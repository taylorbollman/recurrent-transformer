"""Ordinary-autograd FP32 cross-depth read-conditioned memory.

The ordinary early block owns every reader/writer parameter. This module owns
only two adapters, and receives that block functionally on each call. In
particular, slices of its fused QKV projection are tensors, never Parameters.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .exceptions import OLMoConfigurationError


def require_fp32(module: nn.Module, x: torch.Tensor) -> None:
    """Reject silent AMP/dtype changes rather than casting away their evidence."""
    if torch.is_autocast_enabled(x.device.type):
        raise OLMoConfigurationError("CDRM FP32 reference requires autocast disabled")
    if x.dtype != torch.float32 or any(p.dtype != torch.float32 for p in module.parameters()):
        raise OLMoConfigurationError("CDRM FP32 reference requires FP32 activations and parameters")
    if x.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise OLMoConfigurationError("CDRM FP32 reference requires TF32 disabled")


class StatelessRMSNorm(nn.Module):
    """The adapter normalizer; it does not replace the owner's learned norms."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)


def _heads(owner: nn.Module, x: torch.Tensor) -> torch.Tensor:
    b, t, d = x.shape
    return x.reshape(b, t, owner.config.n_heads, d // owner.config.n_heads).transpose(1, 2)


def _clip(owner: nn.Module, x: torch.Tensor) -> torch.Tensor:
    bound = owner.config.clip_qkv
    return x if bound is None else x.clamp(min=-bound, max=bound)


def pre3(owner: nn.Module, p3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-norm, canonical differentiable Q/K/V slices, QK norm, then heads."""
    h = owner.attn_norm(p3)
    qdim, kdim, vdim = owner.fused_dims
    weight, bias = owner.att_proj.weight, owner.att_proj.bias
    q = _clip(owner, F.linear(h, weight[:qdim], None if bias is None else bias[:qdim]))
    kv = _clip(owner, F.linear(h, weight[qdim:], None if bias is None else bias[qdim:]))
    k, v = kv.split((kdim, vdim), dim=-1)
    if owner.q_norm is not None:
        q = owner.q_norm(q)
    if owner.k_norm is not None:
        k = owner.k_norm(k)
    return _heads(owner, q), _heads(owner, k), _heads(owner, v)


def persistent_kv3(owner: nn.Module, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project the residual-space record after its read-conditioned update."""
    h = owner.attn_norm(m)
    qdim, kdim, vdim = owner.fused_dims
    bias = owner.att_proj.bias
    kv = _clip(owner, F.linear(h, owner.att_proj.weight[qdim:], None if bias is None else bias[qdim:]))
    k, v = kv.split((kdim, vdim), dim=-1)
    if owner.k_norm is not None:
        k = owner.k_norm(k)
    return _heads(owner, k), _heads(owner, v)


def read3(owner: nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
          attention_bias: torch.Tensor) -> torch.Tensor:
    """Read an already restricted prefix with its absolute-position bias.

    Explicit FP32 attention scales exactly once. No non-square causal SDPA mask
    is applied: prefix construction has already removed every future record.
    """
    weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + attention_bias, dim=-1)
    attention = weights @ v
    b, _, t, _ = attention.shape
    merged = attention.transpose(1, 2).contiguous().reshape(b, t, owner.config.d_model)
    return owner.attn_out(merged)


def post3(owner: nn.Module, candidate: torch.Tensor, read: torch.Tensor) -> torch.Tensor:
    """Exactly the owner's pre-norm residual/MLP path after projected attention."""
    x = candidate + owner.dropout(read)
    return x + owner.dropout(owner.ff_out(owner.act(owner.ff_proj(owner.ff_norm(x)))))


class CDRMSideMemory(nn.Module):
    """A fresh graph-connected memory is constructed for every independent call."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate_cdrm()
        self.config = config
        self.deep_norm = StatelessRMSNorm(config.cdrm_norm_eps)
        self.bridge_norm = StatelessRMSNorm(config.cdrm_norm_eps)
        self.deep_adapter = nn.Linear(config.d_model, config.d_model, bias=False, device=config.init_device)
        self.bridge_adapter = nn.Linear(config.d_model, config.d_model, bias=False, device=config.init_device)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.config.cdrm_adapter_init_scale / math.sqrt(self.config.d_model)
        nn.init.normal_(self.deep_adapter.weight, std=std)
        nn.init.normal_(self.bridge_adapter.weight, std=std)

    def forward(self, p3: torch.Tensor, p8: torch.Tensor, owner: nn.Module,
                attention_bias: torch.Tensor, output_states: bool = False
                ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        self.config.validate_cdrm()
        require_fp32(self, p3)
        require_fp32(owner, p8)
        if p3.ndim != 3 or p3.shape != p8.shape or p3.shape[-1] != self.config.d_model or p3.shape[1] == 0:
            raise OLMoConfigurationError("CDRM requires matching nonempty [B,T,D] p3/p8 tensors")
        if p3.device != p8.device:
            raise OLMoConfigurationError("CDRM p3 and p8 must share a device")
        if (owner.config.norm_after or owner.config.rope
                or owner.config.effective_n_kv_heads != owner.config.n_heads
                or not hasattr(owner, "att_proj") or owner._activation_checkpoint_fn is not None):
            raise OLMoConfigurationError("CDRM owner must be an ordinary pre-norm full-MHA block without checkpointing")
        b, t, _ = p3.shape
        if (attention_bias is None or attention_bias.dtype != torch.float32 or attention_bias.ndim != 4
                or attention_bias.device != p3.device or attention_bias.shape[0] not in (1, b)
                or attention_bias.shape[1] not in (1, self.config.n_heads)
                or attention_bias.shape[-2:] != (t, t)):
            raise OLMoConfigurationError("CDRM requires an FP32 [1|B,1|H,T,T] absolute-position attention bias")

        source = p8 - p3 if self.config.cdrm_source == "deep" else p3
        deep_correction = self.config.cdrm_epsilon * self.deep_adapter(self.deep_norm(source))
        candidate = p3 + deep_correction
        query, temporary_k, temporary_v = pre3(owner, p3)
        history_k, history_v, proposed, records, read_outputs = [], [], [], [], []

        for token in range(t):
            kt = temporary_k[:, :, token:token + 1]
            vt = temporary_v[:, :, token:token + 1]
            if self.config.cdrm_read_mode == "history":
                k = torch.cat(history_k + [kt], dim=2)
                v = torch.cat(history_v + [vt], dim=2)
                bias = attention_bias[:, :, token:token + 1, :token + 1]
            else:
                k, v = kt, vt
                bias = attention_bias[:, :, token:token + 1, token:token + 1]
            read = read3(owner, query[:, :, token:token + 1], k, v, bias)
            hat_m = post3(owner, candidate[:, token:token + 1], read)
            m = (1.0 - self.config.cdrm_rho) * p3[:, token:token + 1] + self.config.cdrm_rho * hat_m
            # This record first becomes visible on the NEXT iteration. The
            # terminal write is intentionally allowed to have no loss consumer.
            permanent_k, permanent_v = persistent_kv3(owner, m)
            history_k.append(permanent_k)
            history_v.append(permanent_v)
            proposed.append(hat_m)
            records.append(m)
            if output_states:
                read_outputs.append(read)

        hat_m = torch.cat(proposed, dim=1)
        bridge_correction = self.config.cdrm_lambda * self.bridge_adapter(self.bridge_norm(hat_m - p3))
        v8 = p8 + bridge_correction
        states = None
        if output_states:
            states = dict(
                p3=p3, p8=p8, candidate=candidate, reads=torch.cat(read_outputs, dim=1),
                hat_m=hat_m, m=torch.cat(records, dim=1), v8=v8,
                deep_correction=deep_correction, bridge_correction=bridge_correction,
                query=query, temporary_k=temporary_k, temporary_v=temporary_v,
                proposed=tuple(proposed), records=tuple(records),
                permanent_k=tuple(history_k), permanent_v=tuple(history_v),
                read_outputs=tuple(read_outputs),
            )
        return v8, states
