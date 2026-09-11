"""Tiled CDRM contracts: CPU ownership/config and explicitly selected CUDA math.

The FP64 composition lives in test_cdrm_reference and does not call production
Pre/Post, record, attention, side-scan, or adapter-normalization helpers.
"""
import copy
import io
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from olmo.config import ModelConfig
from olmo.exceptions import OLMoConfigurationError
from olmo.model import OLMo
from .test_cdrm_reference import independent_bias, independent_side, nondefault_weights, tiny


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires explicitly selected CUDA container")
ATOL, RTOL = 2e-6, 2e-5


@pytest.fixture(autouse=True)
def deterministic_settings():
    torch.set_num_threads(1)
    torch.manual_seed(7241)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def config(**overrides):
    values = dict(n_layers=5, cdrm_early_layer=1, cdrm_late_layer=3,
                  max_sequence_length=32, cdrm_backend="tiled",
                  cdrm_precision_policy="fp32", reference_eager=True)
    values.update(overrides)
    return tiny(**values)


def model(device="cpu", **overrides):
    if device == "cuda":
        assert Path("/.dockerenv").exists(), "CUDA tests must run inside the project container"
    return nondefault_weights(OLMo(config(**overrides))).to(device)


def assert_close(actual, expected, message="", *, atol=ATOL, rtol=RTOL):
    assert actual is not None and expected is not None, message
    torch.testing.assert_close(actual.detach().double().cpu(), expected.detach().double().cpu(),
                               atol=atol, rtol=rtol, msg=lambda detail: f"{message}: {detail}")


def bias_for(net, value):
    return independent_bias(value.shape[1], net.config.n_heads,
                            net.config.alibi_bias_max, value.dtype).to(value.device)


def side(net, early, late, *, owner=None):
    owner = owner if owner is not None else net.transformer.blocks[net.config.cdrm_early_layer]
    return net.cdrm(early, late, owner, bias_for(net, early), output_states=True)


def side_targets(net, early, late, *, bridge=True):
    owner = net.transformer.blocks[net.config.cdrm_early_layer]
    named = {f"owner/{name}": value for name, value in owner.named_parameters()}
    named["deep_adapter"] = net.cdrm.deep_adapter.weight
    if bridge:
        named["bridge_adapter"] = net.cdrm.bridge_adapter.weight
    named.update(early=early, late=late)
    return {name: value for name, value in named.items() if value.requires_grad}


def test_tiled_configuration_serialization_and_unique_parameter_ownership():
    net = model()
    names = dict(net.named_parameters())
    all_owners = list(net.named_parameters(remove_duplicate=False))
    assert len(names) == len(all_owners) == len({id(p) for _, p in all_owners})
    assert set(dict(net.cdrm.named_parameters())) == {"deep_adapter.weight", "bridge_adapter.weight"}
    assert net.config.cdrm_early_layer == 1 and net.config.cdrm_late_layer == 3
    assert net.config.recurrent_layers == []
    optimizer = torch.optim.AdamW(net.parameters())
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in names.values()}
    stream = io.BytesIO()
    torch.save({"config": asdict(net.config), "model": net.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=False)
    restored = OLMo(ModelConfig(**saved["config"]))
    restored.load_state_dict(saved["model"], strict=True)
    assert restored.config.cdrm_backend == "tiled"
    assert restored.config.cdrm_precision_policy == "fp32"
    for name, value in restored.state_dict().items():
        assert torch.equal(value, net.state_dict()[name]), name


@pytest.mark.parametrize("bad", [
    {"cdrm_backend": "unknown"}, {"cdrm_precision_policy": "unknown"},
    {"cdrm_backend": "naive", "cdrm_precision_policy": "bf16_fp32_state"},
    {"cdrm_rho": 0.0}, {"cdrm_rho": 0.5},
    {"cdrm_source": "same_depth"}, {"cdrm_read_mode": "current_only"},
])
def test_unsupported_tiled_policy_rejected(bad):
    with pytest.raises(OLMoConfigurationError):
        model(**bad)


@CUDA
@pytest.mark.parametrize("length", [1, 5, 16, 17])
def test_tiled_fp32_matches_independent_fp64_side_and_all_gradients(length):
    net = model("cuda")
    early = torch.randn(2, length, 16, device="cuda", requires_grad=True)
    late = torch.randn_like(early, requires_grad=True)
    output, states = side(net, early, late)
    owner = copy.deepcopy(net.transformer.blocks[1]).cpu().double()
    early_ref, late_ref = [v.detach().cpu().double().requires_grad_() for v in (early, late)]
    deep, bridge = [v.detach().cpu().double().requires_grad_() for v in
                    (net.cdrm.deep_adapter.weight, net.cdrm.bridge_adapter.weight)]
    reference = independent_side(owner, early_ref, late_ref, deep, bridge,
        epsilon=net.config.cdrm_epsilon, rho=1.0, bridge_lambda=net.config.cdrm_lambda,
        norm_epsilon=net.config.cdrm_norm_eps)
    for name, ref_name in (("candidate", "candidate"), ("hat_m", "hat_m"),
                           ("m", "memory"), ("v8", "v8")):
        assert_close(states[name], reference[ref_name], name)
    cotangent = torch.randn_like(output)
    named = side_targets(net, early, late)
    gradients = torch.autograd.grad(output, tuple(named.values()), cotangent)
    oracle_named = {f"owner/{name}": value for name, value in owner.named_parameters()}
    oracle_named.update(deep_adapter=deep, bridge_adapter=bridge, early=early_ref, late=late_ref)
    oracle_gradients = torch.autograd.grad(reference["v8"], tuple(oracle_named[name] for name in named), cotangent.cpu().double())
    for name, actual, expected in zip(named, gradients, oracle_gradients):
        assert_close(actual, expected, f"gradient/{name}")
    # autograd.grad must not mutate canonical owners via a nested .backward().
    assert all(parameter.grad is None for parameter in net.parameters())


@CUDA
@pytest.mark.parametrize("length", [1, 5, 16, 17])
def test_causal_side_inputs_fresh_calls_and_batch_isolation(length):
    net = model("cuda")
    early, late = [torch.randn(2, length, 16, device="cuda") for _ in range(2)]
    output, states = side(net, early, late)
    cutoff = max(1, length // 2)
    changed_early, changed_late = early.clone(), late.clone()
    changed_early[:, cutoff:] += 0.7
    changed_late[:, cutoff:] -= torch.linspace(-1, 1, 16, device="cuda")
    altered, changed = side(net, changed_early, changed_late)
    assert_close(output[:, :cutoff], altered[:, :cutoff], "future cannot change past", atol=0, rtol=0)
    assert_close(states["hat_m"][:, :cutoff], changed["hat_m"][:, :cutoff], "causal unscaled branch", atol=0, rtol=0)
    assert_close(side(net, early, late)[0], output, "fresh call", atol=0, rtol=0)
    assert_close(side(net, early[:1], late[:1])[0], output[:1], "example isolation")
    early = early.detach().requires_grad_(); late = late.detach().requires_grad_()
    _, leaf_states = side(net, early, late)
    gradients = torch.autograd.grad(leaf_states["hat_m"][:, :cutoff].square().sum(), (early, late))
    for value in gradients:
        assert torch.count_nonzero(value[:, cutoff:]) == 0


@CUDA
def test_independent_earlier_deep_preview_receives_temporal_credit():
    net = model("cuda", cdrm_epsilon=0.7)
    early, late = [torch.randn(1, 5, 16, device="cuda", requires_grad=True) for _ in range(2)]
    _, states = side(net, early, late)
    direction = torch.randn(16, device="cuda")
    ge, gl = torch.autograd.grad((states["hat_m"][:, -1] * direction).sum(), (early, late))
    assert gl[:, :-1].norm() > 1e-4, "independent earlier deep leaves need future-reader credit"
    assert gl[:, -1].norm() > 1e-4
    assert ge[:, :-1].norm() > 1e-4
    altered = late.detach().clone()
    altered[:, 0] += torch.linspace(-2, 2, 16, device="cuda")
    _, other = side(net, early.detach(), altered)
    assert_close(states["candidate"][:, -1], other["candidate"][:, -1], "unchanged current candidate", atol=0, rtol=0)
    assert (states["hat_m"][:, -1] - other["hat_m"][:, -1]).norm() > 1e-5


@CUDA
def test_lambda_zero_seq_equivalence_and_unscaled_side_stays_testable():
    net = model("cuda", cdrm_lambda=0.0)
    seq = model("cuda", cdrm_enabled=False)
    seq.load_state_dict({name: value for name, value in net.state_dict().items() if not name.startswith("cdrm.")})
    tokens = torch.randint(0, 32, (2, 5), device="cuda")
    with sdpa_kernel(SDPBackend.MATH):
        actual, expected = net(tokens).logits, seq(tokens).logits
    assert_close(actual, expected, "lambda-zero logits", atol=0, rtol=0)
    direction = torch.randn_like(actual)
    shared = dict(net.named_parameters())
    left = torch.autograd.grad(actual, tuple(shared[name] for name, _ in seq.named_parameters()), direction)
    right = torch.autograd.grad(expected, tuple(seq.parameters()), direction)
    for (name, _), a, b in zip(seq.named_parameters(), left, right):
        assert_close(a, b, f"lambda-zero common gradient/{name}")
    early, late = [torch.randn(1, 5, 16, device="cuda", requires_grad=True) for _ in range(2)]
    _, states = side(net, early, late)
    assert states["hat_m"].requires_grad
    grad = torch.autograd.grad(states["hat_m"][:, -1].square().sum(), late)[0]
    assert grad[:, :-1].norm() > 0, "lambda zero must not be used to hide side-gradient defects"


def manual_untied(net, tokens, owner):
    x = net.transformer.wte(tokens)
    if net.config.embedding_layer_norm:
        x = net.transformer.emb_norm(x)
    x = net.transformer.emb_drop(x)
    bias = bias_for(net, x)
    causal = torch.full((tokens.shape[1], tokens.shape[1]), -torch.inf, device=x.device).triu(1)
    bias = bias + causal[None, None]
    for index, block in enumerate(net.transformer.blocks):
        x, _ = block(x, attention_bias=bias)
        if index == net.config.cdrm_early_layer:
            early = x
        if index == net.config.cdrm_late_layer:
            x, _ = net.cdrm(early, x, owner, bias, output_states=False)
    x = net.transformer.ln_f(x)
    return net.transformer.ff_out(x)


@CUDA
def test_shared_owner_autograd_grad_is_preview_plus_independent_side():
    shared = model("cuda")
    untied = copy.deepcopy(shared)
    side_owner = copy.deepcopy(untied.transformer.blocks[1])
    tokens = torch.randint(0, 32, (2, 5), device="cuda")
    with sdpa_kernel(SDPBackend.MATH):
        actual = shared(tokens).logits
        separate = manual_untied(untied, tokens, side_owner)
    assert_close(actual, separate, "tied versus untied forward")
    direction = torch.randn_like(actual)
    named = dict(shared.named_parameters())
    untied_named = dict(untied.named_parameters())
    owner_named = dict(side_owner.named_parameters())
    gradients = dict(zip(named, torch.autograd.grad(actual, tuple(named.values()), direction)))
    values = torch.autograd.grad(separate, tuple(untied_named.values()) + tuple(owner_named.values()), direction)
    preview = dict(zip(untied_named, values[:len(untied_named)]))
    side_grads = dict(zip(owner_named, values[len(untied_named):]))
    for name in named:
        expected = preview[name]
        if name.startswith("transformer.blocks.1."):
            expected = expected + side_grads[name.removeprefix("transformer.blocks.1.")]
        assert_close(gradients[name], expected, f"exactly-once shared ownership/{name}")
    assert all(p.grad is None for p in shared.parameters())
    assert all(p.grad is None for p in untied.parameters())
    assert all(p.grad is None for p in side_owner.parameters())


@CUDA
@pytest.mark.parametrize("freeze_owner,freeze_inputs", [(True, False), (False, True), (True, True)])
def test_frozen_owner_or_inputs_match_naive_autograd(freeze_owner, freeze_inputs):
    actual = model("cuda")
    reference = model("cuda", cdrm_backend="naive")
    reference.load_state_dict(actual.state_dict())
    for net in (actual, reference):
        if freeze_owner:
            net.transformer.blocks[1].requires_grad_(False)
    early, late = [torch.randn(1, 5, 16, device="cuda", requires_grad=not freeze_inputs) for _ in range(2)]
    er, lr = [v.detach().clone().requires_grad_(not freeze_inputs) for v in (early, late)]
    out, _ = side(actual, early, late); ref, _ = side(reference, er, lr)
    direction = torch.randn_like(out)
    targets = side_targets(actual, early, late)
    ref_targets = side_targets(reference, er, lr)
    ga = torch.autograd.grad(out, tuple(targets.values()), direction)
    gr = torch.autograd.grad(ref, tuple(ref_targets.values()), direction)
    for name, a, b in zip(targets, ga, gr):
        assert_close(a, b, f"frozen case/{name}")
    assert all(p.grad is None for p in actual.parameters())


@CUDA
def test_bf16_fixed_forward_scaling_and_unscaled_hat_m_reference():
    mixed = model("cuda", cdrm_precision_policy="bf16_fp32_state", cdrm_lambda=1e-6)
    reference = model("cuda", cdrm_lambda=1e-6)
    reference.load_state_dict(mixed.state_dict())
    early, late = [torch.randn(2, 17, 16, device="cuda", requires_grad=True) for _ in range(2)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, states = side(mixed, early, late)
    _, ref = side(reference, early.detach(), late.detach())
    # The tiny bridge gain cannot conceal an error in the actual recurrent state.
    a, r = states["hat_m"].double(), ref["hat_m"].double()
    floor = ATOL + RTOL * r.abs()
    assert (a-r).norm() <= .03125 * r.norm() + floor.norm()
    assert (a-r).abs().max() <= .0625 * r.abs().max() + floor.max()
    for name in ("candidate", "hat_m", "m", "v8", "deep_correction", "bridge_correction"):
        assert states[name].dtype == torch.float32, name
    direction = torch.randn_like(states["hat_m"])
    targets = side_targets(mixed, early, late, bridge=False)
    base = torch.autograd.grad(states["hat_m"], tuple(targets.values()), direction, retain_graph=True)
    for scale in (1/32, 32.0):
        values = torch.autograd.grad(states["hat_m"], tuple(targets.values()), direction * scale, retain_graph=True)
        for name, expected, value in zip(targets, base, values):
            assert value.dtype == torch.float32 and torch.isfinite(value).all(), name
            assert_close(value.double()/scale, expected.double(), f"fixed-forward scaling/{scale}/{name}", atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in mixed.parameters())
