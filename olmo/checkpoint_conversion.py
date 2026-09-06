"""Exhaustive weights-only conversion between sequential and recurrent OLMo.

Conversion preserves parameter values and ownership, not necessarily the model
function. In particular, removing recurrence is a weight warm start. Optimizer
moments, trainer counters, data position, and recurrence schedules are separate
checkpoint concerns and are deliberately not copied here.
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from .model import OLMo, OLMoRecurrentBlockBase, OLMoSequentialBlock


@dataclass(frozen=True)
class TensorMapping:
    source_keys: Tuple[str, ...]
    target_keys: Tuple[str, ...]
    operation: str
    split_sizes: Tuple[int, ...] = ()


@dataclass
class ConversionReport:
    copied: List[TensorMapping] = field(default_factory=list)
    transformed: List[TensorMapping] = field(default_factory=list)
    new: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    parameter_mappings: List[TensorMapping] = field(default_factory=list)
    source_recurrent_layers: List[int] = field(default_factory=list)
    target_recurrent_layers: List[int] = field(default_factory=list)
    source_key_count: int = 0
    target_key_count: int = 0
    semantics: str = "weights_only"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable conversion audit."""
        return asdict(self)

    def iter_gradient_pairs(
        self, source: OLMo, target: OLMo
    ) -> Iterator[Tuple[str, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Yield every source gradient and the mapped target gradient.

        Split Q/KV target gradients are joined into the source fused-QKV
        layout; the reverse mapping splits a fused target gradient. No gradient
        is silently skipped. Both gradients may be None for an unused parameter,
        but asymmetric missing gradients raise rather than concealing a broken
        gradient path. Compare the yielded pairs under a separately established
        functional-equivalence condition, such as the validated rho-zero case.
        """
        source_params, target_params = dict(source.named_parameters()), dict(target.named_parameters())
        expected_source = {key for mapping in self.parameter_mappings for key in mapping.source_keys}
        expected_target = {key for mapping in self.parameter_mappings for key in mapping.target_keys}
        if set(source_params) != expected_source or set(target_params) != expected_target:
            raise ValueError("Gradient mapping requires the same parameter structures used during conversion")
        for mapping in self.parameter_mappings:
            target_grads = [target_params[key].grad for key in mapping.target_keys]
            if mapping.operation == "split":
                if all(grad is None for grad in target_grads):
                    mapped_grads = [None]
                elif any(grad is None for grad in target_grads):
                    raise ValueError(f"Partially missing split-projection gradients: {mapping.target_keys}")
                else:
                    mapped_grads = [torch.cat(target_grads, dim=0)]
            elif mapping.operation == "concatenate":
                grad = target_grads[0]
                mapped_grads = (
                    [None] * len(mapping.source_keys) if grad is None else grad.split(mapping.split_sizes, dim=0)
                )
            else:
                mapped_grads = target_grads
            for key, mapped_grad in zip(mapping.source_keys, mapped_grads):
                source_grad = source_params[key].grad
                if (source_grad is None) != (mapped_grad is None):
                    raise ValueError(f"Asymmetric missing gradient for {key}")
                yield key, source_grad, mapped_grad


class ConversionError(ValueError):
    """An incompatible or incomplete conversion; target weights were not copied."""

    def __init__(self, message: str, report: Optional[ConversionReport] = None):
        super().__init__(message)
        self.report = report


# These fields change topology, execution policy, or initialization. Every other
# ModelConfig field must agree, including norms, positional encoding, head tying,
# vocabulary and dropout. New config fields are checked by default rather than
# silently accepted by an aging whitelist of semantic fields.
_ALLOWED_CONFIG_DIFFERENCES = {
    "block_type",
    "recurrent_layers",
    "recurrent_backend",
    "recurrent_write_rho",
    "reference_eager",
    "init_device",
    "init_fn",
    "init_std",
    "init_cutoff_factor",
    "scale_emb_init",
    "emb_init_std",
    "precision",
    "bwd_mlp_chunks",
}


def _check_compatibility(source: OLMo, target: OLMo) -> None:
    differences = [
        item.name
        for item in fields(source.config)
        if item.name not in _ALLOWED_CONFIG_DIFFERENCES
        and getattr(source.config, item.name) != getattr(target.config, item.name)
    ]
    if differences:
        raise ConversionError("Incompatible model semantics/configuration: " + ", ".join(differences))
    if source.config.block_group_size != 1 or target.config.block_group_size != 1:
        raise ConversionError("Checkpoint conversion currently requires block_group_size=1")
    for label, model in (("source", source), ("target", target)):
        if len(model.transformer.blocks) != model.config.n_layers:
            raise ConversionError(f"{label} block count disagrees with its configuration")
        for block in model.transformer.blocks:
            if type(block) is not OLMoSequentialBlock and not isinstance(block, OLMoRecurrentBlockBase):
                raise ConversionError(f"Unsupported {label} block class: {type(block).__name__}")
        owners: Dict[int, str] = {}
        storages: Dict[Tuple[str, int], str] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if id(parameter) in owners:
                raise ConversionError(f"{label} has duplicate parameter owners: {owners[id(parameter)]}, {name}")
            owners[id(parameter)] = name
            if parameter.is_meta:
                raise ConversionError(f"Materialize {label} parameter before conversion: {name}")
            if parameter.numel():
                storage = (str(parameter.device), parameter.untyped_storage().data_ptr())
                if storage in storages:
                    raise ConversionError(f"{label} parameters share storage: {storages[storage]}, {name}")
                storages[storage] = name
    for index, (source_block, target_block) in enumerate(
        zip(source.transformer.blocks, target.transformer.blocks)
    ):
        differences = [
            item.name
            for item in fields(source_block.config)
            if item.name not in _ALLOWED_CONFIG_DIFFERENCES
            and getattr(source_block.config, item.name) != getattr(target_block.config, item.name)
        ]
        if differences:
            raise ConversionError(f"Incompatible block {index} semantics/configuration: " + ", ".join(differences))


def convert_model(source: OLMo, target: OLMo) -> ConversionReport:
    """Copy all weights into an independently constructed compatible target.

    All keys, shapes, dtypes, semantics, and ownership are checked before loading
    any target weights. Models must be materialized and use the same weight
    dtype; an explicit precision cast is a separate operation. Legacy duplicate
    helper keys should be loaded through the model's checkpoint migration before
    calling this function. No Parameters or shared source references are created.
    """
    _check_compatibility(source, target)
    source_state, target_state = source.state_dict(), target.state_dict()
    report = ConversionReport(source_key_count=len(source_state), target_key_count=len(target_state))
    for index, (source_block, target_block) in enumerate(
        zip(source.transformer.blocks, target.transformer.blocks)
    ):
        source_rec = isinstance(source_block, OLMoRecurrentBlockBase)
        target_rec = isinstance(target_block, OLMoRecurrentBlockBase)
        if source_rec:
            report.source_recurrent_layers.append(index)
        if target_rec:
            report.target_recurrent_layers.append(index)
        if tuple(source_block.fused_dims) != tuple(target_block.fused_dims):
            raise ConversionError(f"Incompatible fused projection dimensions in block {index}", report)
        if source_rec == target_rec:
            continue
        q_size, k_size, v_size = source_block.fused_dims
        prefix = f"transformer.blocks.{index}."
        for suffix in ("weight", "bias"):
            fused = prefix + f"att_proj.{suffix}"
            split = (prefix + f"q_proj.{suffix}", prefix + f"kv_proj.{suffix}")
            # include_bias=False removes all three tensors.
            if (
                fused not in source_state
                and fused not in target_state
                and not any(key in source_state or key in target_state for key in split)
            ):
                continue
            report.transformed.append(
                TensorMapping(
                    source_keys=split if source_rec else (fused,),
                    target_keys=(fused,) if source_rec else split,
                    operation="concatenate" if source_rec else "split",
                    split_sizes=(q_size, k_size + v_size),
                )
            )
    transformed_sources = {key for mapping in report.transformed for key in mapping.source_keys}
    transformed_targets = {key for mapping in report.transformed for key in mapping.target_keys}
    for key in sorted((set(source_state) & set(target_state)) - transformed_sources - transformed_targets):
        report.copied.append(TensorMapping((key,), (key,), "copy"))
    mappings = report.copied + report.transformed
    accounted_sources = {key for mapping in mappings for key in mapping.source_keys}
    accounted_targets = {key for mapping in mappings for key in mapping.target_keys}
    report.missing = sorted((set(target_state) - accounted_targets) | (accounted_sources - set(source_state)))
    report.unexpected = sorted((set(source_state) - accounted_sources) | (accounted_targets - set(target_state)))
    if report.missing or report.unexpected:
        raise ConversionError(
            f"Incomplete conversion: missing={report.missing}, unexpected={report.unexpected}", report
        )

    converted: Dict[str, torch.Tensor] = {}
    source_params, target_params = dict(source.named_parameters()), dict(target.named_parameters())
    source_storage = {
        (str(parameter.device), parameter.untyped_storage().data_ptr())
        for parameter in source_params.values()
        if parameter.numel()
    }
    if any(
        (str(parameter.device), parameter.untyped_storage().data_ptr()) in source_storage
        for parameter in target_params.values()
        if parameter.numel()
    ):
        raise ConversionError("Source and target must have independent parameter storage", report)

    for mapping in mappings:
        tensors = [source_state[key] for key in mapping.source_keys]
        if mapping.operation == "split":
            tensors = tensors[0].split(mapping.split_sizes, dim=0)
        elif mapping.operation == "concatenate":
            tensors = [torch.cat(tensors, dim=0)]
        for key, tensor in zip(mapping.target_keys, tensors):
            target_tensor = target_state[key]
            if tensor.shape != target_tensor.shape or tensor.dtype != target_tensor.dtype:
                raise ConversionError(
                    f"Shape/dtype mismatch for {key}: {tuple(tensor.shape)}/{tensor.dtype} -> "
                    f"{tuple(target_tensor.shape)}/{target_tensor.dtype}",
                    report,
                )
            if tensor.is_meta or target_tensor.is_meta:
                raise ConversionError(f"Materialize tensors before conversion: {key}", report)
            converted[key] = tensor
        if all(key in source_params for key in mapping.source_keys) and all(
            key in target_params for key in mapping.target_keys
        ):
            report.parameter_mappings.append(mapping)
        elif any(key in source_params for key in mapping.source_keys) or any(
            key in target_params for key in mapping.target_keys
        ):
            raise ConversionError(f"Parameter/buffer ownership mismatch: {mapping}", report)

    removed = set(report.source_recurrent_layers) - set(report.target_recurrent_layers)
    if removed:
        report.semantics = "weight_warm_start"
        report.notes.append(
            f"Recurrence removed at blocks {sorted(removed)}; functional equivalence is not asserted."
        )
    else:
        report.notes.append(
            "Functional equivalence requires separate endpoint tests; weight copying alone does not establish it."
        )
    report.notes.append("Optimizer state, training/data counters, and gate schedules are not converted.")
    # load_state_dict(assign=False) copies into existing target storage and keeps
    # canonical ownership and tied-head semantics established by construction.
    target.load_state_dict(converted, strict=True, assign=False)
    return report
