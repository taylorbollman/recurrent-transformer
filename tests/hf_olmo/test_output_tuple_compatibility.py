"""HF tuple compatibility with native optional diagnostic fields (CPU only)."""

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("with_labels", [False, True])
@pytest.mark.parametrize("with_diagnostics", [False, True])
def test_legacy_tuple_fields_ignore_native_diagnostics(monkeypatch, with_labels, with_diagnostics):
    from hf_olmo import OLMoConfig, OLMoForCausalLM
    from olmo.config import ModelConfig
    from olmo.model import OLMo

    torch.manual_seed(604)
    config = ModelConfig(
        d_model=16, n_heads=4, n_kv_heads=4, n_layers=2, mlp_hidden_size=32,
        vocab_size=32, embedding_size=32, max_sequence_length=8,
        alibi=True, rope=False, recurrent_layers=[], reference_eager=True,
        attention_dropout=0., residual_dropout=0., embedding_dropout=0.,
        init_device="cpu",
    )
    assert config.cdrm_enabled is False
    native = OLMo(config).eval()
    assert not hasattr(native, "cdrm")
    wrapper = OLMoForCausalLM(OLMoConfig(**config.asdict()), model=native).eval()
    native_forward = native.forward
    captured = []

    def forward_with_optional_diagnostics(*args, **kwargs):
        output = native_forward(*args, **kwargs)
        assert output.cdrm_states is None
        if with_diagnostics:
            # Exercise wrapper formatting without claiming HF execution support
            # for a CDRM model or changing the native default-disabled graph.
            output = output._replace(cdrm_states={"test_only_diagnostic": True})
        captured.append(output)
        return output

    monkeypatch.setattr(native, "forward", forward_with_optional_diagnostics)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    labels = tokens.clone() if with_labels else None
    if labels is not None:
        labels[:, 1] = -100
    result = wrapper(tokens, labels=labels, use_cache=True,
                     output_hidden_states=True, return_dict=False)
    assert isinstance(result, tuple)
    assert len(result) == (5 if with_labels else 4)
    output = captured[0]
    legacy = result[1:] if with_labels else result
    assert legacy[0] is output.logits
    assert legacy[1] is output.attn_key_values
    assert legacy[2] is output.hidden_states
    assert legacy[3] is output.pre_logits is None
    assert len(legacy[1]) == 2
    assert len(legacy[2]) == 3
    if labels is not None:
        expected_loss = F.cross_entropy(output.logits[:, :-1].reshape(-1, 32),
                                        labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(result[0], expected_loss)
