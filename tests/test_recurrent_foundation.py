"""Small deterministic NUM fixtures; choose CUDA explicitly with CDRM_TEST_DEVICE.

CPU runs exercise only the eager oracle. GPU runs must use the project container.
No tokenizer, dataset download, training corpus, or external logging is required.
"""

import copy
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from olmo.checkpoint_conversion import convert_model
from olmo.config import ActivationCheckpointingStrategy, ActivationType, BlockType, ModelConfig, TrainConfig
from olmo.efficient_utils import alibi_attention_bias, cuda_capture_block, cuda_capture_model
from olmo.exceptions import OLMoConfigurationError
from olmo.model import OLMo, OLMoRecurrentAutogradBlock, OLMoRecurrentBlockBase, OLMoRecurrentBlockTiled
from olmo.optim import get_param_groups


DEVICE = os.environ.get("CDRM_TEST_DEVICE", "cpu")
ATOL, RTOL = 2e-6, 2e-5


@pytest.fixture(autouse=True)
def deterministic():
    assert DEVICE in ("cpu", "cuda")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if DEVICE == "cuda":
        assert Path("/.dockerenv").is_file(), "GPU tests must run inside the container"
        assert torch.cuda.is_available(), "CUDA requested but unavailable"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    # Tiled helpers legitimately compile several tile sizes/dtypes. Require
    # requested compilation to remain compilation, not a silent cache fallback.
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch.set_num_threads(1)
    torch.manual_seed(937)


def tiny(**overrides):
    values = dict(
        n_layers=12, d_model=32, n_heads=4, n_kv_heads=4, mlp_hidden_size=64,
        activation_type=ActivationType.gelu, vocab_size=32, embedding_size=32,
        eos_token_id=1, pad_token_id=0, max_sequence_length=16, alibi=True, rope=False,
        norm_after=False, attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0,
        include_bias=True, bias_for_layer_norm=True, attention_layer_norm=True,
        embedding_layer_norm=True, weight_tying=False, init_device=DEVICE,
        reference_eager=True,
    )
    values.update(overrides)
    return ModelConfig(**values)


def assert_close(a, b, name="", atol=ATOL, rtol=RTOL):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=lambda m: f"{name}: {m}")


@pytest.mark.parametrize("indices", [[], [3], [8], [3, 8]])
def test_selection_and_independent_configs(indices, tmp_path):
    cfg = tiny(recurrent_layers=indices)
    model = OLMo(cfg)
    assert [i for i, b in enumerate(model.transformer.blocks) if isinstance(b, OLMoRecurrentBlockBase)] == indices
    assert len({id(b.config) for b in model.transformer.blocks}) == 12
    assert cfg.block_type == BlockType.sequential
    assert all(b.config is not cfg for b in model.transformer.blocks)
    cfg.save(tmp_path / "model.yaml")
    assert ModelConfig.load(tmp_path / "model.yaml") == cfg
    if indices:
        model.set_recurrent_write_rho(0.25)
        assert cfg.recurrent_write_rho == 0.25
        assert all(model.transformer.blocks[i].config.recurrent_write_rho == 0.25 for i in indices)


@pytest.mark.parametrize("bad", [
    {"recurrent_layers": [3, 3]}, {"recurrent_layers": [-1]}, {"recurrent_layers": [12]},
    {"recurrent_layers": [True]}, {"recurrent_layers": [3.0]}, {"recurrent_backend": "unknown"},
    {"norm_after": True}, {"rope": True}, {"alibi": False}, {"n_kv_heads": 2},
    {"block_group_size": 2}, {"attention_dropout": 0.1}, {"residual_dropout": 0.1},
    {"embedding_dropout": 0.1}, {"recurrent_write_rho": -0.1}, {"recurrent_write_rho": float("nan")},
    {"recurrent_backend": "tiled", "recurrent_write_rho": 0.5}, {"flash_attention": True},
])
def test_unsupported_config_fails_before_construction(bad):
    config = tiny(recurrent_layers=[3]).update_with(**bad)
    with pytest.raises(OLMoConfigurationError):
        OLMo(config)


@pytest.mark.parametrize("kwargs", [{"use_cache": True}, {"past_key_values": []}, {"doc_lens": torch.tensor([[4]])}])
def test_reject_cached_or_packed_calls(kwargs):
    model = OLMo(tiny(recurrent_layers=[3]))
    with pytest.raises(OLMoConfigurationError):
        model(torch.ones((1, 4), dtype=torch.long, device=DEVICE), **kwargs)


@pytest.mark.parametrize("length", [1, 7])
@pytest.mark.parametrize("clip", [None, 0.07])
def test_rho_zero_random_cotangent_and_ce(length, clip):
    seq = OLMo(tiny(clip_qkv=clip))
    rec = OLMo(tiny(recurrent_layers=[3], recurrent_write_rho=0.0, clip_qkv=clip))
    report = convert_model(seq, rec)
    x = torch.randn(2, length, 32, device=DEVICE, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    logits_seq, logits_rec = seq(input_ids=None, input_embeddings=x).logits, rec(input_ids=None, input_embeddings=y).logits
    assert_close(logits_seq, logits_rec, "logits")
    cotangent = torch.randn_like(logits_seq)
    (logits_seq * cotangent).sum().backward()
    (logits_rec * cotangent).sum().backward()
    assert_close(x.grad, y.grad, "input gradient")
    for name, a, b in report.iter_gradient_pairs(seq, rec):
        if name == "transformer.wte.weight":
            assert a is None and b is None  # Explicit input embeddings bypass wte.
        else:
            assert a is not None and b is not None, name
            assert_close(a, b, name)
    if length > 1:
        targets = torch.randint(0, 32, (2, length - 1), device=DEVICE)
        assert_close(F.cross_entropy(logits_seq[:, :-1].reshape(-1, 32), targets.flatten()),
                     F.cross_entropy(logits_rec[:, :-1].reshape(-1, 32), targets.flatten()), "CE")


def explicit_recurrence(block, x, bias):
    """Independent functional oracle: no Pre/Post module calls or mutable cache."""
    cfg = block.config
    bsz, length, width = x.shape
    hd = width // cfg.n_heads
    def heads(t):
        return t.reshape(bsz, -1, cfg.n_heads, hd).transpose(1, 2)
    normed = block.attn_norm(x)
    q, kv = block.q_proj(normed), block.kv_proj(normed)
    if cfg.clip_qkv is not None:
        q, kv = q.clamp(-cfg.clip_qkv, cfg.clip_qkv), kv.clamp(-cfg.clip_qkv, cfg.clip_qkv)
    k, v = kv.split(block.fused_dims[1:], dim=-1)
    if block.q_norm is not None:
        q, k = block.q_norm(q), block.k_norm(k)
    q, k, v = heads(q), heads(k), heads(v)
    history_k, history_v, outputs = [], [], []
    for t in range(length):
        kt = torch.cat(history_k + [k[:, :, t:t+1]], dim=2)
        vt = torch.cat(history_v + [v[:, :, t:t+1]], dim=2)
        scores = q[:, :, t:t+1] @ kt.transpose(-1, -2) / hd**0.5
        attention = scores.add(bias[:, :, t:t+1, :t+1]).softmax(-1) @ vt
        attention = block.attn_out(attention.transpose(1, 2).reshape(bsz, 1, width))
        residual = x[:, t:t+1] + attention
        out = residual + block.ff_out(block.act(block.ff_proj(block.ff_norm(residual))))
        outputs.append(out)
        write = (1 - cfg.recurrent_write_rho) * x[:, t:t+1] + cfg.recurrent_write_rho * out
        kv_write = block.kv_proj(block.attn_norm(write))
        if cfg.clip_qkv is not None:
            kv_write = kv_write.clamp(-cfg.clip_qkv, cfg.clip_qkv)
        kw, vw = kv_write.split(block.fused_dims[1:], dim=-1)
        if block.k_norm is not None:
            kw = block.k_norm(kw)
        history_k.append(heads(kw))
        history_v.append(heads(vw))
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("rho", [0.0, 0.37, 1.0])
def test_independent_block_oracle(rho):
    model = OLMo(tiny(recurrent_layers=[3], recurrent_write_rho=rho, clip_qkv=0.12))
    block = model.transformer.blocks[3]
    other = copy.deepcopy(block)
    x = torch.randn(2, 7, 32, device=DEVICE, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    bias = alibi_attention_bias(model, TrainConfig(model=model.config), 7)
    actual = block(x, attention_bias=bias)[0]
    expected = explicit_recurrence(other, y, bias)
    assert_close(actual, expected, "oracle output")
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)
    assert_close(x.grad, y.grad, "oracle input gradient")
    for name, param in block.named_parameters():
        assert param.grad is not None, name
        assert_close(param.grad, dict(other.named_parameters())[name].grad, name)


@pytest.mark.parametrize("indices", [[], [3], [3, 8]])
def test_whole_model_causality_and_padding(indices):
    model = OLMo(tiny(recurrent_layers=indices)).eval()
    tokens = torch.randint(1, 32, (2, 9), device=DEVICE)
    changed = tokens.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 32
    mask = torch.ones_like(tokens)
    mask[:, 7:] = 0
    with torch.no_grad():
        original = model(tokens, attention_mask=mask).logits
        perturbed = model(changed, attention_mask=mask).logits
        prefix = model(tokens[:, :5]).logits
    assert_close(original[:, :5], perturbed[:, :5], "future perturbation", atol=0, rtol=0)
    assert_close(original[:, :5], prefix, "prefix consistency")


def test_raw_alibi_direct_block_is_causal_and_capture_rejected():
    model = OLMo(tiny())
    block = model.transformer.blocks[0]
    x = torch.randn(2, 7, 32, device=DEVICE)
    y = x.clone()
    y[:, 4:] += torch.randn_like(y[:, 4:])
    raw_bias = model.get_alibi_attention_bias(7, torch.device(DEVICE))
    with torch.no_grad():
        assert_close(block(x, raw_bias)[0][:, :4], block(y, raw_bias)[0][:, :4], atol=0, rtol=0)
    cfg = TrainConfig(model=model.config)
    bias = alibi_attention_bias(model, cfg, 7)
    assert (bias[0, 0].triu(1)[torch.ones(7, 7, dtype=torch.bool, device=DEVICE).triu(1)] < -1e20).all()
    with pytest.raises(OLMoConfigurationError, match="capture"):
        cuda_capture_model(model, cfg)
    with pytest.raises(OLMoConfigurationError, match="capture"):
        cuda_capture_block(block, cfg, bias)


def test_write_order_temporal_gradient_and_single_token():
    model = OLMo(tiny(recurrent_layers=[3]))
    block = model.transformer.blocks[3]
    captured = []
    def remember(module, inputs, output):
        captured.append(output)
        output.retain_grad()
    handle = block.post_attention_block.register_forward_hook(remember)
    x = torch.randn(2, 5, 32, device=DEVICE, requires_grad=True)
    bias = alibi_attention_bias(model, TrainConfig(model=model.config), 5)
    result = block(x, bias)[0]
    assert len(captured) == 5
    result[:, -1].square().sum().backward()
    assert captured[0].grad is not None and captured[0].grad.abs().max() > 1e-8
    handle.remove()
    model.set_recurrent_write_rho(0)
    zero = block(x[:, :1], bias[:, :, :1, :1])[0]
    model.set_recurrent_write_rho(1)
    one = block(x[:, :1], bias[:, :, :1, :1])[0]
    assert_close(zero, one, "T=1", atol=0, rtol=0)
    # Corrupt only persistent projections: the current position cannot read it.
    calls = []
    def corrupt_persistent(module, inputs, output):
        calls.append(1)
        return output if len(calls) == 1 else output * 0 + 9
    with torch.no_grad():
        baseline = block(x.detach(), bias)[0]
        hook = block.kv_proj.register_forward_hook(corrupt_persistent)
        perturbed = block(x.detach(), bias)[0]
        hook.remove()
    assert len(calls) == 6  # one temporary batch + one persistent projection per position
    assert_close(baseline[:, 0], perturbed[:, 0], atol=0, rtol=0)
    assert not torch.allclose(baseline[:, 1:], perturbed[:, 1:])


def test_canonical_ownership_legacy_aliases_and_optimizer():
    model = OLMo(tiny(recurrent_layers=[3]))
    block = model.transformer.blocks[3]
    assert not list(block.pre_attention_block.parameters())
    assert not list(block.post_attention_block.parameters())
    assert block.pre_attention_block.q_proj is block.q_proj
    assert block.post_attention_block.ff_norm is block.ff_norm
    names = list(model.state_dict())
    assert not any("pre_attention_block" in name or "post_attention_block" in name for name in names)
    cfg = TrainConfig(model=model.config)
    groups = get_param_groups(cfg, model)
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    state = copy.deepcopy(model.state_dict())
    canonical = "transformer.blocks.3.attn_norm.weight"
    alias = "transformer.blocks.3.pre_attention_block.attn_norm.weight"
    state[alias] = state[canonical].clone()
    model.load_state_dict(state, strict=True)
    state[alias] += 1
    with pytest.raises(RuntimeError, match="conflicting legacy alias"):
        model.load_state_dict(state, strict=True)


def test_naive_whole_layer_checkpointing():
    model = OLMo(tiny(recurrent_layers=[3]))
    checkpointed = copy.deepcopy(model)
    checkpointed.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    x = torch.randn(2, 7, 32, device=DEVICE, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = model(input_ids=None, input_embeddings=x).logits, checkpointed(input_ids=None, input_embeddings=y).logits
    cotangent = torch.randn_like(a)
    a.backward(cotangent)
    b.backward(cotangent)
    assert_close(x.grad, y.grad)
    for name, p in model.named_parameters():
        other = dict(checkpointed.named_parameters())[name]
        if p.grad is not None:
            assert_close(p.grad, other.grad, name)


@pytest.mark.gpu
@pytest.mark.skipif(DEVICE != "cuda", reason="explicit GPU execution required")
@pytest.mark.parametrize("length", [1, 7, 16])
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("chunks", [1, 4])
def test_tiled_random_cotangent(length, eager, chunks):
    model = OLMo(tiny(recurrent_layers=[3]))
    tiled = OLMo(tiny(recurrent_layers=[3], recurrent_backend="tiled", reference_eager=eager, bwd_mlp_chunks=chunks))
    report = convert_model(model, tiled)
    x = torch.randn(2, length, 32, device=DEVICE, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = model(input_ids=None, input_embeddings=x).logits, tiled(input_ids=None, input_embeddings=y).logits
    assert_close(a, b, "tiled logits")
    cotangent = torch.randn_like(a)
    a.backward(cotangent)
    b.backward(cotangent)
    assert_close(x.grad, y.grad, "tiled input gradients")
    for name, pa, pb in report.iter_gradient_pairs(model, tiled):
        if name == "transformer.wte.weight":
            assert pa is pb is None
        else:
            assert pa is not None and pb is not None, name
            assert_close(pa, pb, name)
    with pytest.raises(OLMoConfigurationError):
        tiled.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)


@pytest.mark.gpu
@pytest.mark.skipif(DEVICE != "cuda", reason="explicit GPU execution required")
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("chunks", [1, 4])
def test_bf16_tiled_random_cotangent(eager, chunks):
    # BF16 is a separate numerical contract; FP32 checks above remain authoritative.
    naive = OLMo(tiny(recurrent_layers=[3], include_bias=False, bias_for_layer_norm=False))
    tiled = OLMo(tiny(recurrent_layers=[3], recurrent_backend="tiled", reference_eager=eager,
                     bwd_mlp_chunks=chunks, include_bias=False, bias_for_layer_norm=False))
    report = convert_model(naive, tiled)
    tokens = torch.randint(1, 32, (2, 9), device=DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = naive(tokens).logits, tiled(tokens).logits
    assert_close(a, b, "BF16 logits", atol=2e-3, rtol=2e-2)
    cotangent = torch.randn_like(a)
    fp32_logits = naive(tokens).logits
    fp32_logits.backward(cotangent.float())
    fp32_grads = {name: p.grad.detach().clone() for name, p in naive.named_parameters()}
    naive.zero_grad(set_to_none=True)
    a.backward(cotangent)
    b.backward(cotangent)
    for name, pa, pb in report.iter_gradient_pairs(naive, tiled):
        assert pa is not None and pb is not None, name
        # Fixed absolute gradient tolerances ignore tensor scale and cancellation.
        # Bound every parameter separately in units of BF16 epsilon, AND compare
        # both backends with the same FP32 reference/cotangent. No aggregate can
        # conceal a missing or inaccurate parameter gradient. See the preserved
        # bf16_diagnostic report for the rejected initial atol=.003/rtol=.03 test.
        eps = torch.finfo(torch.bfloat16).eps
        for label, ref, actual in (("naive vs FP32", fp32_grads[name], pa),
                                   ("tiled vs FP32", fp32_grads[name], pb),
                                   ("tiled vs naive", pa, pb)):
            assert torch.isfinite(actual).all(), name
            scale = ref.double().square().mean().sqrt().clamp_min(1e-12)
            diff = (ref - actual).double()
            relative_l2 = diff.norm() / ref.double().norm().clamp_min(1e-12)
            assert relative_l2 <= 2 * eps, (label, name, "relative L2", relative_l2.item())
            assert diff.abs().max() / scale <= 8 * eps, (label, name, "max/RMS", (diff.abs().max()/scale).item())
