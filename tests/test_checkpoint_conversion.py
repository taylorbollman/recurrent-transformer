"""Small CPU fixtures for exhaustive conversion and its gradient audit."""

import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from olmo.checkpoint_conversion import ConversionError, convert_model
from olmo.config import BlockType, ModelConfig
from olmo.model import OLMo


def _config(**overrides):
    values = dict(
        d_model=16,
        n_heads=4,
        n_layers=12,
        mlp_hidden_size=32,
        vocab_size=32,
        embedding_size=32,
        max_sequence_length=8,
        block_type=BlockType.sequential,
        recurrent_layers=[],
        recurrent_backend="naive",
        recurrent_write_rho=1.0,
        reference_eager=True,
        alibi=True,
        rope=False,
        flash_attention=False,
        norm_after=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        include_bias=True,
        bias_for_layer_norm=True,
        embedding_layer_norm=True,
        attention_layer_norm=True,
        weight_tying=False,
        init_device="cpu",
    )
    values.update(overrides)
    return ModelConfig(**values)


def _nondefault_weights(model):
    generator = torch.Generator().manual_seed(482)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name or "ln_f" in name:
                parameter.uniform_(0.7, 1.3, generator=generator)
                if name.endswith("bias"):
                    parameter.mul_(0.1)
            else:
                parameter.normal_(std=0.07, generator=generator)


@pytest.mark.parametrize("weight_tying", [False, True])
@pytest.mark.parametrize("include_bias", [False, True])
def test_exhaustive_conversion_and_serialization(tmp_path, weight_tying, include_bias):
    config = _config(weight_tying=weight_tying, include_bias=include_bias)
    source = OLMo(config)
    _nondefault_weights(source)
    target_config = replace(config, recurrent_layers=[3], recurrent_write_rho=0.0)
    target = OLMo(target_config)
    target_parameter_ids = {name: id(parameter) for name, parameter in target.named_parameters()}

    report = convert_model(source, target)
    assert report.new == report.missing == report.unexpected == []
    assert report.source_recurrent_layers == []
    assert report.target_recurrent_layers == [3]
    assert len(report.transformed) == (2 if include_bias else 1)
    assert {key for mapping in report.copied + report.transformed for key in mapping.source_keys} == set(
        source.state_dict()
    )
    assert {key for mapping in report.copied + report.transformed for key in mapping.target_keys} == set(
        target.state_dict()
    )
    assert {key for mapping in report.parameter_mappings for key in mapping.source_keys} == set(
        dict(source.named_parameters())
    )
    assert {key for mapping in report.parameter_mappings for key in mapping.target_keys} == set(
        dict(target.named_parameters())
    )
    assert target_parameter_ids == {name: id(parameter) for name, parameter in target.named_parameters()}
    assert len(list(target.named_parameters(remove_duplicate=False))) == len(target_parameter_ids)
    assert not any("pre_attention_block." in key or "post_attention_block." in key for key in target.state_dict())
    assert ("ff_out" not in target.transformer) == weight_tying
    assert json.loads(json.dumps(report.to_dict()))["missing"] == []

    source_state, target_state = source.state_dict(), target.state_dict()
    for key in source_state.keys() & target_state.keys():
        torch.testing.assert_close(target_state[key], source_state[key], rtol=0, atol=0)
    for suffix in ("weight", "bias") if include_bias else ("weight",):
        fused = source_state[f"transformer.blocks.3.att_proj.{suffix}"]
        split = torch.cat(
            (
                target_state[f"transformer.blocks.3.q_proj.{suffix}"],
                target_state[f"transformer.blocks.3.kv_proj.{suffix}"],
            ),
            dim=0,
        )
        torch.testing.assert_close(split, fused, rtol=0, atol=0)

    checkpoint = tmp_path / "converted.pt"
    torch.save(target.state_dict(), checkpoint)
    loaded = OLMo(target_config)
    loaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    for key, value in target_state.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)

    # Removing recurrence is explicitly labeled a warm start, even though this
    # particular rho-zero fixture has a separately testable equivalent function.
    restored = OLMo(config)
    reverse_report = convert_model(loaded, restored)
    assert reverse_report.semantics == "weight_warm_start"
    for key, value in source_state.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    # Loading copies into existing storage; later target updates cannot mutate
    # either the source model or the source checkpoint tensors.
    before = source.transformer.wte.weight.detach().clone()
    with torch.no_grad():
        target.transformer.wte.weight.add_(2)
    torch.testing.assert_close(source.transformer.wte.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("norm_after", True),
        ("layer_norm_eps", 0.1),
        ("clip_qkv", 0.4),
        ("weight_tying", True),
        ("attention_dropout", 0.2),
        ("alibi_bias_max", 4.0),
        ("embedding_layer_norm", False),
        ("attention_layer_norm", False),
    ],
)
def test_rejects_incompatible_source_semantics_without_mutation(field, value):
    config = _config()
    source = OLMo(replace(config, **{field: value}))
    target = OLMo(replace(config, recurrent_layers=[3]))
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(ConversionError, match=field):
        convert_model(source, target)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("extra_owner,report_field", [("source", "unexpected"), ("target", "missing")])
def test_unexplained_key_is_an_error_with_audit(extra_owner, report_field):
    source, target = OLMo(_config()), OLMo(_config(recurrent_layers=[3]))
    (source if extra_owner == "source" else target).register_buffer("unexplained", torch.ones(1))
    before = target.transformer.wte.weight.detach().clone()
    with pytest.raises(ConversionError, match="Incomplete conversion") as failure:
        convert_model(source, target)
    assert getattr(failure.value.report, report_field) == ["unexplained"]
    torch.testing.assert_close(target.transformer.wte.weight, before, rtol=0, atol=0)


def test_rejects_implicit_dtype_conversion_and_duplicate_parameter_ownership():
    source, target = OLMo(_config()), OLMo(_config(recurrent_layers=[3]))
    target.double()
    with pytest.raises(ConversionError, match="Shape/dtype mismatch"):
        convert_model(source, target)
    target.float()
    target.duplicate = target.transformer.wte.weight
    with pytest.raises(ConversionError, match="duplicate parameter owners"):
        convert_model(source, target)
    del target.duplicate
    target.shared_storage = torch.nn.Parameter(target.transformer.wte.weight[:1])
    with pytest.raises(ConversionError, match="parameters share storage"):
        convert_model(source, target)
    with pytest.raises(ConversionError, match="independent parameter storage"):
        convert_model(source, source)


def test_checks_actual_per_block_configurations():
    source, target = OLMo(_config()), OLMo(_config(recurrent_layers=[3]))
    # Model-level fields agree, but a per-block semantic change must still fail.
    source.transformer.blocks[3].config.clip_qkv = 0.2
    with pytest.raises(ConversionError, match="block 3.*clip_qkv"):
        convert_model(source, target)


def test_full_gradient_audit_in_both_projection_directions():
    torch.manual_seed(91)
    # Four blocks retain the block-3 conversion while keeping this CPU fixture
    # small; the state coverage fixture above uses the full twelve-block layout.
    config = _config(n_layers=4)
    source, target = OLMo(config), OLMo(replace(config, recurrent_layers=[3], recurrent_write_rho=0.0))
    _nondefault_weights(source)
    report = convert_model(source, target)
    tokens = torch.randint(config.vocab_size, (2, 5))
    for model in (source, target):
        logits = model(tokens).logits
        F.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1)).backward()
    pairs = list(report.iter_gradient_pairs(source, target))
    assert len(pairs) == len(list(source.parameters()))
    assert any(name == "transformer.blocks.3.att_proj.weight" for name, _, _ in pairs)
    for name, source_grad, target_grad in pairs:
        assert source_grad is not None, name
        torch.testing.assert_close(source_grad, target_grad, rtol=2e-4, atol=2e-6, msg=name)

    reverse = convert_model(target, source)
    # Weights were already corresponding and existing gradients are retained.
    for name, target_grad, source_grad in reverse.iter_gradient_pairs(target, source):
        torch.testing.assert_close(target_grad, source_grad, rtol=2e-4, atol=2e-6, msg=name)
    target.transformer.blocks[3].kv_proj.weight.grad = None
    with pytest.raises(ValueError, match="Partially missing"):
        list(report.iter_gradient_pairs(source, target))


def test_legacy_debug_entrypoint_uses_exhaustive_converter():
    from debug_utils import initialize_recurrent_from_sequential

    source, target = OLMo(_config()), OLMo(_config(recurrent_layers=[3]))
    _nondefault_weights(source)
    report = initialize_recurrent_from_sequential(source, target)
    assert report.missing == report.unexpected == []
    torch.testing.assert_close(source.transformer.ln_f.weight, target.transformer.ln_f.weight, rtol=0, atol=0)
    torch.testing.assert_close(source.transformer.emb_norm.bias, target.transformer.emb_norm.bias, rtol=0, atol=0)
