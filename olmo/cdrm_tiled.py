"""Tiled rho-one CDRM scan with explicit, functional gradient ownership.

Candidate/QKV preprocessing and the late bridge stay in ordinary autograd. This
function owns no parameters: canonical parameters of the ordinary early block
are tensor inputs, and its backward returns their side-scan contributions.
Only first derivatives are supported. Projected persistent records are saved
in linear memory; attention histories are reconstructed during backward.
"""
from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable

from .exceptions import OLMoConfigurationError


def _observe(owner, phase, tensors, token_index=None):
    callback = getattr(owner, "_cdrm_precision_observer", None)
    if callback is not None:
        callback(phase, tensors, token_index=token_index)


def _call(owner, function, *arguments):
    if owner.config.reference_eager:
        function = getattr(function, "_torchdynamo_orig_callable", function)
    return function(*arguments)


@torch.compile(dynamic=False)
def _post_body(candidate, attention, owner, mixed, cache_enabled):
    from .cdrm import post3
    with torch.autocast("cuda", enabled=mixed, dtype=torch.bfloat16, cache_enabled=cache_enabled):
        attention = attention.to(torch.bfloat16 if mixed else torch.float32)
        read = owner.attn_out(attention)
        return post3(owner, candidate, read)


@torch.compile(dynamic=False)
def _writer_body(records, owner, mixed, cache_enabled):
    from .cdrm import persistent_kv3
    with torch.autocast("cuda", enabled=mixed, dtype=torch.bfloat16, cache_enabled=cache_enabled):
        return persistent_kv3(owner, records)


class _TiledScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, candidate, query, temporary_k, temporary_v, attention_bias, owner, *parameters):
        # These are captured before disabling AMP for all recurrent arithmetic.
        mixed = torch.is_autocast_enabled("cuda")
        cache_enabled = torch.is_autocast_cache_enabled()
        with torch.autocast("cuda", enabled=False):
            return _TiledScan._forward(ctx, candidate, query, temporary_k, temporary_v,
                                       attention_bias, owner, parameters, mixed, cache_enabled)

    @staticmethod
    def _forward(ctx, candidate, query, temporary_k, temporary_v, attention_bias,
                 owner, parameters, mixed, cache_enabled):
        # Lazy import avoids the model -> cdrm -> tiled -> model import cycle.
        from .model import block_attention_add

        batch, length, width = candidate.shape
        heads = owner.config.n_heads
        head_width = width // heads
        scale = 1.0 / math.sqrt(head_width)
        q_math = query.float().permute(2, 0, 1, 3).contiguous() * scale
        k_math = temporary_k.float().permute(2, 0, 1, 3)
        # clone is necessary: singleton B/H dimensions can make this permutation
        # contiguous already, so contiguous() alone could alias an input tensor.
        weighted_values = temporary_v.float().permute(2, 0, 1, 3).contiguous().clone()
        max_logit = (k_math * q_math).sum(dim=-1)
        max_logit = max_logit + attention_bias.diagonal(dim1=-2, dim2=-1).expand(batch, heads, length).permute(2, 0, 1)
        sum_scores = torch.ones_like(max_logit)
        final_k = torch.empty((length, batch, heads, head_width), dtype=query.dtype, device=query.device)
        final_v = torch.empty_like(final_k)
        outputs = []
        _observe(owner, "forward.projected", {"candidate": candidate, "query": query,
                                             "temporary_k": temporary_k, "temporary_v": temporary_v})
        _observe(owner, "forward.initial_state", {"q_math": q_math, "weighted_values": weighted_values,
                                                 "max_logit": max_logit, "sum_scores": sum_scores})
        for token in range(length):
            attention = weighted_values[token:token + 1] / sum_scores[token:token + 1].unsqueeze(-1)
            merged = attention.transpose(0, 1).reshape(batch, 1, width)
            current = candidate[:, token:token + 1]
            _observe(owner, "forward.attention", {"attention": attention}, token)
            _observe(owner, "forward.post_input", {"candidate": current, "attention": merged}, token)
            proposed = _call(owner, _post_body, current, merged, owner, mixed, cache_enabled)
            outputs.append(proposed)
            # rho=1: the proposed state is exactly the next permanent record.
            key, value = _call(owner, _writer_body, proposed, owner, mixed, cache_enabled)
            final_k[token:token + 1] = key.permute(2, 0, 1, 3)
            final_v[token:token + 1] = value.permute(2, 0, 1, 3)
            _observe(owner, "forward.output", {"hat_m": proposed}, token)
            _observe(owner, "forward.permanent_storage", {"k": key, "v": value}, token)
            if token + 1 == length:
                break
            tile = (token + 1) & -(token + 1)
            written = slice(token + 1 - tile, token + 1)
            future = slice(token + 1, min(token + 1 + tile, length))
            weighted_values[future], max_logit[future], sum_scores[future] = _call(
                owner, block_attention_add, weighted_values[future], max_logit[future], sum_scores[future],
                final_k[written].permute(1, 2, 0, 3), final_v[written].permute(1, 2, 0, 3),
                q_math[future].permute(1, 2, 0, 3), attention_bias[:, :, future, written], True)
        result = torch.cat(outputs, dim=1)
        ctx.save_for_backward(candidate, query, temporary_k, temporary_v, result, final_k, final_v,
                              attention_bias, *parameters)
        ctx.owner = owner
        ctx.parameter_names = tuple(name for name, _ in owner.named_parameters())
        ctx.mixed = mixed
        ctx.cache_enabled = cache_enabled
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        with torch.autocast("cuda", enabled=False):
            return _TiledScan._backward(ctx, grad_output)

    @staticmethod
    def _backward(ctx, grad_output):
        from .model import recompute_alphas, recompute_atts

        candidate, query, temporary_k, temporary_v, outputs, final_k_saved, final_v_saved, bias, *parameters = ctx.saved_tensors
        owner, mixed, cache_enabled = ctx.owner, ctx.mixed, ctx.cache_enabled
        current_parameters = dict(owner.named_parameters())
        if tuple(current_parameters) != ctx.parameter_names or any(
                current_parameters[name] is not parameter for name, parameter in zip(ctx.parameter_names, parameters)):
            raise RuntimeError("CDRM owner parameters changed between forward and backward")
        batch, length, width = candidate.shape
        heads, head_width = owner.config.n_heads, width // owner.config.n_heads
        scale = 1.0 / math.sqrt(head_width)
        q = query.detach().float().permute(2, 0, 1, 3).contiguous() * scale
        k_init = temporary_k.detach().float().permute(2, 0, 1, 3).contiguous()
        v_init = temporary_v.detach().float().permute(2, 0, 1, 3).contiguous()
        final_k, final_v = final_k_saved.float(), final_v_saved.float()
        alphas = _call(owner, recompute_alphas, final_k, k_init, q, bias, True)
        atts = _call(owner, recompute_atts, final_v, v_init, alphas, True)
        k_grads, v_grads = torch.zeros_like(final_k), torch.zeros_like(final_v)
        gs = torch.empty_like(q)
        g_dot_atts = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        all_grads = torch.empty_like(candidate)
        final_v_head_major = final_v.permute(1, 2, 0, 3).contiguous()
        _observe(owner, "backward.recomputed_attention", {"alphas": alphas, "attention": atts})
        _observe(owner, "backward.buffers", {"k_grads": k_grads, "v_grads": v_grads,
                                            "gs": gs, "g_dot_atts": g_dot_atts})

        for token in range(length - 1, -1, -1):
            # Writer VJP computes only the record-input adjoint here. Parameter
            # contributions are computed once in the final batched writer pass.
            record = outputs[:, token:token + 1].detach().requires_grad_(True)
            with torch.enable_grad():
                key, value = _call(owner, _writer_body, record, owner, mixed, cache_enabled)
                key_math = key.float().permute(2, 0, 1, 3)
                value_math = value.float().permute(2, 0, 1, 3)
                writer_credit = torch.autograd.grad(
                    (key_math, value_math), record,
                    grad_outputs=(k_grads[token:token + 1], v_grads[token:token + 1]))[0]
            total_credit = grad_output[:, token:token + 1].float() + writer_credit
            all_grads[:, token:token + 1] = total_credit
            attention_leaf = atts[token:token + 1].detach().requires_grad_(True)
            with torch.enable_grad():
                merged = attention_leaf.transpose(0, 1).reshape(batch, 1, width)
                proposed = _call(owner, _post_body, candidate[:, token:token + 1].detach(),
                                 merged, owner, mixed, cache_enabled)
                attention_credit = torch.autograd.grad(proposed, attention_leaf,
                                                        grad_outputs=total_credit)[0]
            gs[token:token + 1] = attention_credit
            g_dot_atts[token:token + 1] = (atts[token:token + 1] * attention_credit).sum(dim=-1)
            _observe(owner, "backward.writer_replay", {"k": key, "v": value}, token)
            _observe(owner, "backward.local_post", {"hat_m": proposed, "attention": merged}, token)
            _observe(owner, "backward.temporal_adjoint", {"writer_credit": writer_credit,
                "total_hat_m_credit": total_credit, "attention_credit": attention_credit}, token)
            if token == 0:
                continue
            tile = token & -token
            earlier = slice(token - tile, token)
            later = slice(token, min(token + tile, length))
            if tile > 1:
                v_grads[earlier] += torch.matmul(alphas[:, :, earlier, later],
                    gs[later].permute(1, 2, 0, 3)).permute(2, 0, 1, 3)
                alphas[:, :, earlier, later] *= (
                    torch.matmul(final_v_head_major[:, :, earlier], gs[later].permute(1, 2, 3, 0))
                    - g_dot_atts[later].permute(1, 2, 0).unsqueeze(-2))
                k_grads[earlier] += torch.matmul(alphas[:, :, earlier, later],
                    q[later].permute(1, 2, 0, 3)).permute(2, 0, 1, 3)
            else:
                v_grads[token - 1] += alphas[:, :, token - 1, token:token + 1] * gs[token]
                alphas[:, :, token - 1, token] *= (
                    (final_v_head_major[:, :, token - 1] * gs[token]).sum(dim=-1) - g_dot_atts[token])
                k_grads[token - 1] += alphas[:, :, token - 1, token:token + 1] * q[token]

        diagonal = alphas.diagonal(dim1=2, dim2=3).permute(2, 0, 1)
        v_init_grad = diagonal.unsqueeze(-1) * gs
        self_logits_grad = diagonal * ((v_init * gs).sum(dim=-1) - g_dot_atts)
        k_init_grad = self_logits_grad.unsqueeze(-1) * q
        q_grad = self_logits_grad.unsqueeze(-1) * k_init
        alphas.diagonal(dim1=2, dim2=3).zero_()
        q_grad += torch.matmul(alphas.permute(0, 1, 3, 2), final_k.permute(1, 2, 0, 3)).permute(2, 0, 1, 3)

        # Only these autograd.grad results contribute owner parameter gradients;
        # local temporal VJPs above never write any parameter .grad field.
        active = [(index, parameter) for index, parameter in enumerate(parameters)
                  if ctx.needs_input_grad[6 + index]]
        parameter_grads = [None] * len(parameters)
        candidate_leaf = candidate.detach().requires_grad_(True)
        with torch.enable_grad():
            merged = atts.detach().transpose(0, 1).reshape(batch, length, width)
            proposed = _call(owner, _post_body, candidate_leaf, merged, owner, mixed, cache_enabled)
            post_grads = torch.autograd.grad(proposed, (candidate_leaf, *(parameter for _, parameter in active)),
                                            grad_outputs=all_grads, allow_unused=True)
        candidate_grad = post_grads[0]
        for (index, _), gradient in zip(active, post_grads[1:]):
            parameter_grads[index] = gradient

        writer_active = [(index, parameter) for index, parameter in active
                         if ctx.parameter_names[index].startswith(("att_proj.", "attn_norm.", "k_norm."))]
        if writer_active:
            with torch.enable_grad():
                key, value = _call(owner, _writer_body, outputs.detach(), owner, mixed, cache_enabled)
                writer_grads = torch.autograd.grad(
                    (key.float(), value.float()), tuple(parameter for _, parameter in writer_active),
                    grad_outputs=(k_grads.permute(1, 2, 0, 3), v_grads.permute(1, 2, 0, 3)), allow_unused=True)
            for (index, _), gradient in zip(writer_active, writer_grads):
                if gradient is not None:
                    parameter_grads[index] = gradient if parameter_grads[index] is None else parameter_grads[index] + gradient

        q_grad = (q_grad * scale).permute(1, 2, 0, 3).to(query.dtype)
        k_init_grad = k_init_grad.permute(1, 2, 0, 3).to(temporary_k.dtype)
        v_init_grad = v_init_grad.permute(1, 2, 0, 3).to(temporary_v.dtype)
        _observe(owner, "backward.dense_replay", {"hat_m": proposed, "total_hat_m_credit": all_grads})
        _observe(owner, "backward.input_adjoints", {"candidate": candidate_grad, "query": q_grad,
                                                    "temporary_k": k_init_grad, "temporary_v": v_init_grad})
        input_grads = (candidate_grad, q_grad, k_init_grad, v_init_grad)
        return (*(gradient if ctx.needs_input_grad[index] else None for index, gradient in enumerate(input_grads)),
                None, None, *parameter_grads)


def tiled_scan(candidate, query, temporary_k, temporary_v, owner, attention_bias):
    """Return rho-one read-conditioned proposed records, with first derivatives.

BF16 calls require outer CUDA BF16 autocast. Recurrent arithmetic and residual
records remain FP32. Frozen inputs/owner parameters are supported independently;
the custom function never relies on candidate.requires_grad to collect weights.
"""
    if candidate.device.type != "cuda":
        raise OLMoConfigurationError("Tiled CDRM requires CUDA; use the naive FP32 reference on CPU")
    if candidate.ndim != 3 or candidate.shape[1] < 1 or candidate.dtype != torch.float32:
        raise OLMoConfigurationError("Tiled CDRM candidate must be nonempty FP32 [B,T,D]")
    batch, length, width = candidate.shape
    if width != owner.config.d_model or width % owner.config.n_heads:
        raise OLMoConfigurationError("Tiled CDRM width/head dimensions differ from the owner")
    expected = (batch, owner.config.n_heads, length, width // owner.config.n_heads)
    mixed = torch.is_autocast_enabled("cuda")
    if mixed and torch.get_autocast_dtype("cuda") != torch.bfloat16:
        raise OLMoConfigurationError("Tiled CDRM supports FP32 or CUDA BF16 autocast only")
    expected_policy = "bf16_fp32_state" if mixed else "fp32"
    if owner.config.cdrm_precision_policy != expected_policy:
        raise OLMoConfigurationError("Tiled CDRM owner precision policy must match the explicit outer autocast context")
    if owner.config.cdrm_rho != 1.0 or owner.config.cdrm_read_mode != "history" or owner.config.cdrm_source != "deep":
        raise OLMoConfigurationError("Tiled CDRM supports rho=1, history reads and deep candidates only")
    dtype = torch.bfloat16 if mixed else torch.float32
    if any(value.shape != expected or value.device != candidate.device or value.dtype != dtype
           for value in (query, temporary_k, temporary_v)):
        raise OLMoConfigurationError("Tiled CDRM Q/K/V must match owner shape and active projection dtype")
    if (attention_bias is None or attention_bias.ndim != 4 or attention_bias.dtype != torch.float32
            or attention_bias.device != candidate.device or attention_bias.shape[0] not in (1, batch)
            or attention_bias.shape[1] not in (1, owner.config.n_heads)
            or attention_bias.shape[-2:] != (length, length) or attention_bias.requires_grad):
        raise OLMoConfigurationError("Tiled CDRM requires a fixed FP32 causal/ALiBi attention bias")
    if (owner.config.norm_after or owner.config.rope or owner.config.effective_n_kv_heads != owner.config.n_heads
            or owner.config.attention_dropout or owner.config.residual_dropout or owner.config.embedding_dropout
            or owner._activation_checkpoint_fn is not None or not hasattr(owner, "att_proj")):
        raise OLMoConfigurationError("Tiled CDRM owner must be an ordinary pre-norm full-MHA block without dropout/checkpointing")
    parameters = tuple(owner.parameters())
    if any(parameter.device != candidate.device or parameter.dtype != torch.float32 for parameter in parameters):
        raise OLMoConfigurationError("Tiled CDRM canonical owner parameters must remain FP32 on the candidate device")
    return _TiledScan.apply(candidate, query, temporary_k, temporary_v, attention_bias, owner, *parameters)
