"""Independent ordinary-autograd CDRM reference contracts (CPU, no GPU fallback)."""

import copy
import io
import math
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from olmo.config import ActivationType, ModelConfig, TrainConfig
from olmo.exceptions import OLMoConfigurationError
from olmo.model import OLMo, OLMoSequentialBlock
from olmo.optim import get_param_groups


ATOL, RTOL = 2e-6, 2e-5


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(7241)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


def independent_bias(length, heads, maximum=8.0, dtype=torch.float64):
    """Absolute-position ALiBi, not the production bias/helper or SDPA mask."""
    query = torch.arange(length, dtype=dtype)[:, None]
    key = torch.arange(length, dtype=dtype)[None, :]
    slopes = 2.0 ** (-maximum * torch.arange(1, heads + 1, dtype=dtype) / heads)
    return -slopes[None, :, None, None] * (query-key).abs()[None, None]


def functional_norm(x, module):
    return F.layer_norm(x, module.normalized_shape, module.weight, module.bias, module.eps)


def functional_linear(x, module):
    return F.linear(x, module.weight, module.bias)


def independent_side(block, p3, p8, deep_weight, bridge_weight, *, epsilon, rho,
                     bridge_lambda, norm_epsilon, same_depth=False, current_only=False):
    """Independent explicit FP64 composition with canonical fused weight slices.

    No production side-scan, Pre/Post, persistent-record, adapter-normalizer,
    attention-bias, or SDPA helper is invoked. Functional primitives still
    differentiate the owning parameters, including fused QKV and learned norms.
    """
    batch, length, width = p3.shape
    heads = block.config.n_heads
    head_dim = width // heads
    qdim, kdim, vdim = block.fused_dims
    bias = independent_bias(length, heads, block.config.alibi_bias_max, p3.dtype)

    def split_heads(x):
        return x.reshape(batch, -1, heads, head_dim).transpose(1, 2)

    def rms_norm(x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_epsilon)

    def project(x, start, size):
        weight = block.att_proj.weight[start:start+size]
        bias_value = None if block.att_proj.bias is None else block.att_proj.bias[start:start+size]
        result = F.linear(x, weight, bias_value)
        if block.config.clip_qkv is not None:
            result = result.clamp(-block.config.clip_qkv, block.config.clip_qkv)
        return result

    normed = functional_norm(p3, block.attn_norm)
    q = project(normed, 0, qdim)
    k = project(normed, qdim, kdim)
    v = project(normed, qdim+kdim, vdim)
    if block.q_norm is not None:
        q, k = functional_norm(q, block.q_norm), functional_norm(k, block.k_norm)
    q, k, v = map(split_heads, (q, k, v))
    deep_input = p3 if same_depth else p8-p3
    candidate = p3 + epsilon * F.linear(rms_norm(deep_input), deep_weight)
    history_k, history_v, proposed, records, reads = [], [], [], [], []
    for t in range(length):
        keys = torch.cat(([] if current_only else history_k)+[k[:, :, t:t+1]], dim=2)
        values = torch.cat(([] if current_only else history_v)+[v[:, :, t:t+1]], dim=2)
        scores = torch.einsum('bhqd,bhkd->bhqk', q[:, :, t:t+1], keys) / math.sqrt(head_dim)
        scores = scores + (bias[:, :, t:t+1, t:t+1] if current_only else bias[:, :, t:t+1, :t+1])
        probabilities = scores.softmax(-1)
        read = torch.einsum('bhqk,bhkd->bhqd', probabilities, values)
        read = functional_linear(read.transpose(1, 2).reshape(batch, 1, width), block.attn_out)
        residual = candidate[:, t:t+1] + read
        hidden = functional_norm(residual, block.ff_norm)
        hat_m = residual + functional_linear(F.gelu(functional_linear(hidden, block.ff_proj), approximate='none'), block.ff_out)
        memory = (1-rho)*p3[:, t:t+1] + rho*hat_m
        write_norm = functional_norm(memory, block.attn_norm)
        wk, wv = project(write_norm, qdim, kdim), project(write_norm, qdim+kdim, vdim)
        if block.k_norm is not None:
            wk = functional_norm(wk, block.k_norm)
        history_k.append(split_heads(wk))
        history_v.append(split_heads(wv))
        reads.append(read)
        proposed.append(hat_m)
        records.append(memory)
    hat_m = torch.cat(proposed, dim=1)
    memory = torch.cat(records, dim=1)
    correction = bridge_lambda * F.linear(rms_norm(hat_m-p3), bridge_weight)
    return dict(candidate=candidate, reads=torch.cat(reads, dim=1), hat_m=hat_m,
                memory=memory, correction=correction, v8=p8+correction,
                keys=history_k, values=history_v)


def tiny(**overrides):
    values = dict(
        n_layers=12, d_model=16, n_heads=4, n_kv_heads=4, mlp_hidden_size=32,
        activation_type=ActivationType.gelu, vocab_size=32, embedding_size=32,
        eos_token_id=1, pad_token_id=0, max_sequence_length=16, alibi=True, rope=False,
        norm_after=False, attention_dropout=0.0, residual_dropout=0.0,
        embedding_dropout=0.0, include_bias=True, bias_for_layer_norm=True,
        attention_layer_norm=True, embedding_layer_norm=True, weight_tying=False,
        init_device="cpu", reference_eager=True, recurrent_layers=[],
        cdrm_enabled=True, cdrm_epsilon=0.35, cdrm_lambda=0.2, cdrm_rho=1.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


def nondefault_weights(model):
    """Exercise learned norms and every bias; avoid zero-adapter degeneracy."""
    generator = torch.Generator().manual_seed(821)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name or "ln_f" in name:
                parameter.uniform_(0.7, 1.3, generator=generator)
                if name.endswith("bias"):
                    parameter.mul_(0.1)
            else:
                parameter.normal_(std=0.13, generator=generator)
    return model


def make_model(**overrides):
    return nondefault_weights(OLMo(tiny(**overrides)))


def close(actual, expected, name="", *, atol=ATOL, rtol=RTOL):
    torch.testing.assert_close(actual.double(), expected.double(), atol=atol, rtol=rtol,
                               msg=lambda message: f"{name}: {message}")


def side(model, p3, p8, *, owner=None):
    if owner is None:
        owner = model.transformer.blocks[model.config.cdrm_early_layer]
    return model.cdrm(p3, p8, owner,
                      independent_bias(p3.shape[1], model.config.n_heads,
                                       model.config.alibi_bias_max, p3.dtype),
                      output_states=True)


def manual_with_side_owner(model, tokens, owner):
    """Untied diagnostic graph: preview owner and side owner are independent."""
    x = model.transformer.wte(tokens)
    if model.config.embedding_layer_norm:
        x = model.transformer.emb_norm(x)
    x = model.transformer.emb_drop(x)
    length = tokens.shape[1]
    bias = independent_bias(length, model.config.n_heads,
                            model.config.alibi_bias_max, x.dtype)
    causal = torch.full((length, length), float("-inf"), dtype=x.dtype).triu(1)
    bias = bias + causal[None, None]
    for index, block in enumerate(model.transformer.blocks):
        x, _ = block(x, attention_bias=bias)
        if index == model.config.cdrm_early_layer:
            p3 = x
        if index == model.config.cdrm_late_layer:
            x, _ = model.cdrm(p3, x, owner, bias, output_states=False)
    x = model.transformer.ln_f(x)
    logits = F.linear(x, model.transformer.wte.weight) if model.config.weight_tying else model.transformer.ff_out(x)
    return logits / math.sqrt(model.config.d_model) if model.config.scale_logits else logits


@pytest.mark.parametrize("rho,source,read_mode", [(1.0, "deep", "history"),
                                                (0.0, "deep", "history"),
                                                (0.4, "same_depth", "history"),
                                                (1.0, "deep", "current_only")])
@pytest.mark.parametrize("width,heads", [(16, 4), (32, 16)])
def test_explicit_fp64_oracle_all_side_gradients(rho, source, read_mode, width, heads):
    model = make_model(cdrm_rho=rho, cdrm_source=source, cdrm_read_mode=read_mode,
                       d_model=width, n_heads=heads, n_kv_heads=heads)
    p3 = torch.randn(2, 5, width, requires_grad=True)
    p8 = torch.randn(2, 5, width, requires_grad=True)
    actual, states = side(model, p3, p8)
    oracle_owner = copy.deepcopy(model.transformer.blocks[3]).double()
    deep = model.cdrm.deep_adapter.weight.detach().double().clone().requires_grad_()
    bridge = model.cdrm.bridge_adapter.weight.detach().double().clone().requires_grad_()
    p3_ref, p8_ref = (value.detach().double().requires_grad_() for value in (p3, p8))
    reference = independent_side(
        oracle_owner, p3_ref, p8_ref, deep, bridge, epsilon=model.config.cdrm_epsilon,
        rho=rho, bridge_lambda=model.config.cdrm_lambda,
        norm_epsilon=model.config.cdrm_norm_eps, same_depth=source == "same_depth",
        current_only=read_mode == "current_only",
    )
    for name, reference_name in (("candidate", "candidate"), ("hat_m", "hat_m"),
                                  ("m", "memory"), ("v8", "v8")):
        close(states[name], reference[reference_name], name)
    for name, reference_name in (("permanent_k", "keys"), ("permanent_v", "values")):
        for index, (value, expected) in enumerate(zip(states[name], reference[reference_name])):
            close(value, expected, f"{name}/{index}")
    cotangent = torch.randn_like(actual)
    # This scalar touches the side output; the last persistent write is correctly unused.
    (actual * cotangent).sum().backward()
    (reference["v8"] * cotangent.double()).sum().backward()
    close(p3.grad, p3_ref.grad, "p3 gradient")
    close(p8.grad, p8_ref.grad, "p8 gradient")
    close(model.cdrm.deep_adapter.weight.grad, deep.grad, "deep adapter gradient")
    close(model.cdrm.bridge_adapter.weight.grad, bridge.grad, "bridge adapter gradient")
    expected_parameters = dict(oracle_owner.named_parameters())
    for name, parameter in model.transformer.blocks[3].named_parameters():
        assert parameter.grad is not None, name
        close(parameter.grad, expected_parameters[name].grad, f"side owner {name}")


@pytest.mark.parametrize("weight_tying", [False, True])
def test_lambda_zero_matches_seq_logits_and_all_common_gradients(weight_tying):
    active = make_model(cdrm_lambda=0.0, weight_tying=weight_tying)
    sequential = OLMo(tiny(cdrm_enabled=False, weight_tying=weight_tying))
    common = {name: value for name, value in active.state_dict().items() if not name.startswith("cdrm.")}
    sequential.load_state_dict(common, strict=True)
    tokens = torch.randint(0, 32, (2, 7))
    labels = torch.randint(0, 32, (2, 7))
    labels[:, :3] = -100
    # OLMo construction enables SDPA backends, so select math after both models
    # exist. CDRM's ordinary blocks also enforce this policy internally.
    with sdpa_kernel(SDPBackend.MATH):
        actual = active(tokens).logits
        expected = sequential(tokens).logits
    close(actual, expected, "lambda zero logits", atol=0, rtol=0)
    for logits in (actual, expected):
        F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100).backward()
    active_parameters = dict(active.named_parameters())
    for name, parameter in sequential.named_parameters():
        assert parameter.grad is not None, name
        close(active_parameters[name].grad, parameter.grad, f"lambda zero {name}")
    for name, parameter in active.cdrm.named_parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0, name


def test_preview_is_ordinary_once_and_hidden_state_nine_is_bridge():
    model = make_model()
    visits, outputs = [], {}
    handles = []
    for index, block in enumerate(model.transformer.blocks):
        assert isinstance(block, OLMoSequentialBlock)
        def record(_module, _inputs, output, index=index):
            visits.append(index)
            outputs[index] = output[0]
        handles.append(block.register_forward_hook(record))
    result = model(torch.randint(0, 32, (2, 7)), output_hidden_states=True,
                   output_cdrm_states=True)
    for handle in handles:
        handle.remove()
    assert visits == list(range(12))
    assert len(result.hidden_states) == 13
    assert result.cdrm_states["p3"] is outputs[3]
    assert result.cdrm_states["p8"] is outputs[8]
    assert result.hidden_states[9] is result.cdrm_states["v8"]
    assert result.cdrm_states["hat_m"].grad_fn is not None
    assert model(torch.randint(0, 32, (1, 1))).cdrm_states is None


@pytest.mark.parametrize("rho", [0.0, 0.4, 1.0])
def test_non_power_of_two_causality_and_batch_memory_isolation(rho):
    model = make_model(cdrm_rho=rho)
    tokens = torch.randint(0, 32, (2, 7))
    original = model(tokens, output_cdrm_states=True)
    changed = tokens.clone()
    changed[:, 3:] = (changed[:, 3:] + 9) % 32
    perturbed = model(changed, output_cdrm_states=True)
    close(original.logits[:, :3], perturbed.logits[:, :3], "causal logits", atol=0, rtol=0)
    for name in ("p3", "p8", "candidate", "hat_m", "m", "v8"):
        close(original.cdrm_states[name][:, :3], perturbed.cdrm_states[name][:, :3], name, atol=0, rtol=0)
    # An unrelated call cannot mutate a carried memory bank, and examples cannot share one.
    repeated = model(tokens).logits
    close(repeated, original.logits, "no cross-call memory", atol=0, rtol=0)
    close(model(tokens[:1]).logits, original.logits[:1], "no cross-example memory")
    embeddings = model.transformer.wte(tokens).detach().requires_grad_()
    model(None, input_embeddings=embeddings).logits[:, :3].sum().backward()
    assert torch.count_nonzero(embeddings.grad[:, 3:]) == 0


def test_single_token_read_precedes_write_and_bridge_uses_proposed_memory():
    zero, one = make_model(cdrm_rho=0.0), make_model(cdrm_rho=1.0)
    p3, p8 = torch.randn(2, 1, 16), torch.randn(2, 1, 16)
    output_zero, state_zero = side(zero, p3, p8)
    output_one, state_one = side(one, p3, p8)
    close(output_zero, output_one, "single-token rho invariance", atol=0, rtol=0)
    close(state_zero["hat_m"], state_one["hat_m"], "read before write", atol=0, rtol=0)
    close(state_zero["m"], p3, "rho zero anchor", atol=0, rtol=0)
    close(state_one["m"], state_one["hat_m"], "rho one memory", atol=0, rtol=0)
    assert (output_zero-p8).norm() > 0.01, "rho zero must keep the active side bridge"
    assert not torch.equal(state_zero["m"], state_one["m"])


@pytest.mark.parametrize("rho", [0.0, 1.0])
def test_independent_deep_leaf_temporal_write_credit(rho):
    model = make_model(cdrm_rho=rho, cdrm_epsilon=0.7, cdrm_lambda=0.4)
    p3 = torch.randn(1, 5, 16, requires_grad=True)
    p8 = torch.randn(1, 5, 16, requires_grad=True)
    _, states = side(model, p3, p8)
    # Independent leaf p3 excludes the ordinary preview shortcut from earlier p8.
    direction = torch.randn(16)
    (states["bridge_correction"][:, -1] * direction).sum().backward()
    assert p8.grad[:, -1].norm() > 1e-4, "the current deep candidate remains active"
    if rho == 0:
        assert torch.count_nonzero(p8.grad[:, :-1]) == 0
    else:
        assert p8.grad[:, :-1].norm() > 1e-4, "earlier deep state must receive later memory credit"
    assert p3.grad[:, :-1].norm() > 1e-4


def test_writes_are_conditioned_on_history_read_not_only_local_candidate():
    model = make_model(cdrm_epsilon=0.7, cdrm_rho=1.0)
    p3, p8 = torch.randn(1, 5, 16), torch.randn(1, 5, 16)
    changed = p8.clone()
    changed[:, 0] += torch.linspace(-2, 2, 16)
    _, first = side(model, p3, p8)
    _, second = side(model, p3, changed)
    for name in ("candidate", "query", "temporary_k", "temporary_v"):
        # Q/K/V are head-major; candidate is sequence-major.
        left = first[name][:, -1] if name == "candidate" else first[name][:, :, -1]
        right = second[name][:, -1] if name == "candidate" else second[name][:, :, -1]
        close(left, right, f"unchanged local {name}", atol=0, rtol=0)
    for name in ("reads", "hat_m", "m"):
        assert (first[name][:, -1]-second[name][:, -1]).norm() > 1e-5, name
    assert (first["permanent_v"][-1]-second["permanent_v"][-1]).norm() > 1e-5


def test_shared_owner_gradient_equals_preview_plus_untied_side_gradient():
    shared = make_model()
    untied = copy.deepcopy(shared)
    side_owner = copy.deepcopy(untied.transformer.blocks[3])
    tokens = torch.randint(0, 32, (2, 5))
    actual = shared(tokens).logits
    diagnostic = manual_with_side_owner(untied, tokens, side_owner)
    close(actual, diagnostic, "shared versus untied forward")
    cotangent = torch.randn_like(actual)
    (actual*cotangent).sum().backward()
    (diagnostic*cotangent).sum().backward()
    shared_params = dict(shared.named_parameters())
    separate = dict(side_owner.named_parameters())
    checked = []
    for name, parameter in untied.named_parameters():
        expected = parameter.grad
        assert expected is not None, name
        if name.startswith("transformer.blocks.3."):
            local_name = name.removeprefix("transformer.blocks.3.")
            side_gradient = separate[local_name].grad
            assert side_gradient is not None and side_gradient.norm() > 0, local_name
            assert expected.norm() > 0, local_name
            expected = expected + side_gradient
            checked.append(local_name)
        close(shared_params[name].grad, expected, f"shared gradient {name}")
    assert set(checked) == set(separate)
    for required in ("att_proj.weight", "att_proj.bias", "attn_norm.weight", "attn_norm.bias",
                     "q_norm.weight", "q_norm.bias", "k_norm.weight", "k_norm.bias",
                     "ff_norm.weight", "ff_norm.bias"):
        assert required in checked
    qdim, kdim, vdim = side_owner.fused_dims
    for gradient in (side_owner.att_proj.weight.grad, untied.transformer.blocks[3].att_proj.weight.grad):
        assert all(piece.norm() > 0 for piece in gradient.split((qdim, kdim, vdim), dim=0))


@pytest.mark.parametrize("weight_tying", [False, True])
def test_unique_optimizer_ownership_and_serialization(weight_tying):
    model = make_model(weight_tying=weight_tying)
    registered = list(model.named_parameters(remove_duplicate=False))
    assert len({id(parameter) for _, parameter in registered}) == len(registered)
    assert len({parameter.untyped_storage().data_ptr() for _, parameter in registered}) == len(registered)
    side_names = {name for name, _ in model.cdrm.named_parameters()}
    assert side_names == {"deep_adapter.weight", "bridge_adapter.weight"}
    assert model.cdrm.deep_adapter.bias is model.cdrm.bridge_adapter.bias is None
    assert all(parameter.norm() > 0 for parameter in model.cdrm.parameters())
    optimizer_groups = get_param_groups(TrainConfig(model=model.config), model)
    optimized = [parameter for group in optimizer_groups for parameter in group["params"]]
    assert len(optimized) == len(registered)
    assert {id(parameter) for parameter in optimized} == {id(parameter) for _, parameter in registered}
    saved = io.BytesIO()
    torch.save(dict(config=asdict(model.config), model=model.state_dict()), saved)
    saved.seek(0)
    packet = torch.load(saved, map_location="cpu", weights_only=False)
    restored = OLMo(ModelConfig(**packet["config"]))
    restored.load_state_dict(packet["model"], strict=True)
    assert restored.config == model.config
    tokens = torch.randint(0, 32, (2, 7))
    close(restored(tokens).logits, model(tokens).logits, "serialized CDRM", atol=0, rtol=0)
    assert not ({id(p) for p in restored.parameters()} & {id(p) for p in model.parameters()})


def test_terminal_write_has_no_consumer_but_earlier_records_do():
    model = make_model()
    p3 = torch.randn(1, 5, 16, requires_grad=True)
    p8 = torch.randn(1, 5, 16, requires_grad=True)
    output, states = side(model, p3, p8)
    records = states["records"]
    gradients = torch.autograd.grad(output[:, -1].square().sum(), records, allow_unused=True)
    assert all(gradient is not None and gradient.norm() > 0 for gradient in gradients[:-1])
    assert gradients[-1] is None or torch.count_nonzero(gradients[-1]) == 0


def test_constructor_preserves_paired_backbone_and_initializes_active_adapters():
    torch.manual_seed(662)
    sequential = OLMo(tiny(cdrm_enabled=False))
    torch.manual_seed(662)
    active = OLMo(tiny())
    parameters = dict(active.named_parameters())
    for name, parameter in sequential.named_parameters():
        close(parameters[name], parameter, f"paired initialization {name}", atol=0, rtol=0)
    for adapter in (active.cdrm.deep_adapter, active.cdrm.bridge_adapter):
        assert adapter.weight.requires_grad and adapter.weight.norm() > 0
        assert adapter.bias is None
    for normalizer in (active.cdrm.deep_norm, active.cdrm.bridge_norm):
        assert not list(normalizer.parameters())
        assert not list(normalizer.buffers())
        assert normalizer.eps == active.config.cdrm_norm_eps


def test_adam_state_and_next_update_survive_serialization():
    original = make_model()
    optimizer = torch.optim.AdamW(get_param_groups(TrainConfig(model=original.config), original),
                                  lr=5e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0)
    tokens, labels = torch.randint(0, 32, (2, 5)), torch.randint(0, 32, (2, 5))
    def update(model, opt):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(tokens).logits.flatten(0, 1), labels.flatten()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    update(original, optimizer)
    saved = io.BytesIO()
    torch.save(dict(model=original.state_dict(), optimizer=optimizer.state_dict()), saved)
    saved.seek(0)
    packet = torch.load(saved, map_location="cpu", weights_only=False)
    restored = OLMo(copy.deepcopy(original.config))
    restored.load_state_dict(packet["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(get_param_groups(TrainConfig(model=restored.config), restored),
                                           lr=5e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0)
    restored_optimizer.load_state_dict(packet["optimizer"])
    update(original, optimizer)
    update(restored, restored_optimizer)
    for name, value in original.state_dict().items():
        close(value, restored.state_dict()[name], f"resumed parameter {name}", atol=0, rtol=0)
    for index, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            close(value, restored_optimizer.state_dict()["state"][index][name],
                  f"resumed Adam {index}/{name}", atol=0, rtol=0)


@pytest.mark.parametrize("bad", [
    {"recurrent_layers": [3]}, {"cdrm_backend": "tiled"}, {"cdrm_rho": -0.1},
    {"cdrm_rho": 1.1}, {"cdrm_lambda": float("nan")}, {"cdrm_epsilon": float("inf")},
    {"cdrm_early_layer": 8}, {"cdrm_late_layer": 11}, {"cdrm_source": "unknown"},
    {"cdrm_read_mode": "unknown"}, {"cdrm_norm_eps": 0.0}, {"cdrm_adapter_init_scale": 0.0},
    {"n_kv_heads": 2}, {"norm_after": True}, {"alibi": False}, {"rope": True},
    {"attention_dropout": 0.1}, {"residual_dropout": 0.1}, {"embedding_dropout": 0.1},
    {"precision": "amp_bf16"}, {"block_group_size": 2}, {"flash_attention": True},
])
def test_unsupported_configuration_fails_closed(bad):
    with pytest.raises(OLMoConfigurationError):
        OLMo(tiny(**bad))


@pytest.mark.parametrize("mode", ["cache", "packing", "mask", "bias", "autocast", "bf16"])
def test_unsupported_forward_fails_closed(mode):
    model = make_model()
    tokens = torch.randint(0, 32, (1, 5))
    kwargs = {}
    if mode == "cache":
        kwargs["use_cache"] = True
    elif mode == "packing":
        kwargs.update(doc_lens=torch.tensor([[5]]), max_doc_lens=[5])
    elif mode == "mask":
        kwargs["attention_mask"] = torch.ones_like(tokens)
    elif mode == "bias":
        kwargs["attention_bias"] = torch.zeros(1, 1, 5, 5)
    elif mode == "bf16":
        model.bfloat16()
    with pytest.raises(OLMoConfigurationError):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mode == "autocast"):
            model(tokens, **kwargs)


@pytest.mark.parametrize("late", [3, 4])
def test_six_block_topology_uses_configured_sites_and_canonical_owner(late):
    model = make_model(n_layers=6, cdrm_early_layer=1, cdrm_late_layer=late)
    visits, outputs, side_inputs = [], {}, []
    handles = []
    for index, block in enumerate(model.transformer.blocks):
        assert isinstance(block, OLMoSequentialBlock)
        def record(_module, _inputs, output, index=index):
            visits.append(index)
            outputs[index] = output[0]
        handles.append(block.register_forward_hook(record))
    handles.append(model.cdrm.register_forward_pre_hook(lambda _module, inputs: side_inputs.append(inputs)))
    tokens = torch.randint(0, 32, (2, 7))
    result = model(tokens, output_hidden_states=True, output_cdrm_states=True)
    for handle in handles:
        handle.remove()
    assert visits == list(range(6)), "Every ordinary block must execute exactly once"
    assert len(side_inputs) == 1
    assert side_inputs[0][0] is outputs[1]
    assert side_inputs[0][1] is outputs[late]
    assert side_inputs[0][2] is model.transformer.blocks[1]
    # These public diagnostic names refer to the configured early/late states;
    # they do not require physical block indices 3 and 8.
    assert result.cdrm_states["p3"] is outputs[1]
    assert result.cdrm_states["p8"] is outputs[late]
    assert len(result.hidden_states) == 7
    assert result.hidden_states[late+1] is result.cdrm_states["v8"]
    assert (result.cdrm_states["v8"]-outputs[late]).norm() > 0.01
    changed = tokens.clone()
    changed[:, 3:] = (changed[:, 3:]+9) % 32
    close(model(changed).logits[:, :3], result.logits[:, :3],
          "six-block causal prefix", atol=0, rtol=0)


@pytest.mark.parametrize("late", [3, 4])
def test_six_block_lambda_zero_matches_seq_and_all_common_gradients(late):
    active = make_model(n_layers=6, cdrm_early_layer=1, cdrm_late_layer=late, cdrm_lambda=0.0)
    sequential = OLMo(tiny(n_layers=6, cdrm_enabled=False,
                           cdrm_early_layer=1, cdrm_late_layer=late))
    sequential.load_state_dict({name: value for name, value in active.state_dict().items()
                                if not name.startswith("cdrm.")}, strict=True)
    tokens, labels = torch.randint(0, 32, (2, 7)), torch.randint(0, 32, (2, 7))
    labels[:, :3] = -100
    with sdpa_kernel(SDPBackend.MATH):
        actual, expected = active(tokens).logits, sequential(tokens).logits
    close(actual, expected, "six-block lambda-zero logits", atol=0, rtol=0)
    actual_loss = F.cross_entropy(actual.flatten(0, 1), labels.flatten(), ignore_index=-100)
    expected_loss = F.cross_entropy(expected.flatten(0, 1), labels.flatten(), ignore_index=-100)
    close(actual_loss, expected_loss, "six-block lambda-zero loss", atol=0, rtol=0)
    actual_loss.backward()
    expected_loss.backward()
    active_parameters = dict(active.named_parameters())
    for name, parameter in sequential.named_parameters():
        assert parameter.grad is not None, name
        close(active_parameters[name].grad, parameter.grad, f"six-block lambda-zero {name}")
    for name, parameter in active.cdrm.named_parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0, name


@pytest.mark.parametrize("late", [3, 4])
@pytest.mark.parametrize("rho", [0.0, 1.0])
def test_six_block_independent_deep_credit_uses_early_owner(late, rho):
    model = make_model(n_layers=6, cdrm_early_layer=1, cdrm_late_layer=late,
                       cdrm_rho=rho, cdrm_epsilon=0.7, cdrm_lambda=0.4)
    early = torch.randn(1, 5, 16, requires_grad=True)
    deep = torch.randn(1, 5, 16, requires_grad=True)
    _, states = side(model, early, deep)
    direction = torch.randn(16)
    (states["bridge_correction"][:, -1]*direction).sum().backward()
    assert deep.grad[:, -1].norm() > 1e-4
    if rho == 0:
        assert torch.count_nonzero(deep.grad[:, :-1]) == 0
    else:
        assert deep.grad[:, :-1].norm() > 1e-4
    assert early.grad[:, :-1].norm() > 1e-4
    owner = model.transformer.blocks[model.config.cdrm_early_layer]
    assert all(parameter.grad is not None for parameter in owner.parameters())
    for index, block in enumerate(model.transformer.blocks):
        if index != model.config.cdrm_early_layer:
            assert all(parameter.grad is None for parameter in block.parameters()), index
    qdim, kdim, vdim = owner.fused_dims
    assert all(piece.norm() > 0 for piece in owner.att_proj.weight.grad.split((qdim, kdim, vdim), dim=0))
