"""Single-model text DPO helpers; imports and validation never allocate a GPU."""
from __future__ import annotations

import inspect
import math


def validate_preference_records(records: list[dict]) -> None:
    if not records:
        raise ValueError("DPO requires nonempty preference records")
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            raise ValueError("DPO requires records with explicit approval metadata")
        metadata = row.get("metadata", {})
        split = str(metadata.get("split") or metadata.get("partition") or row.get("split") or "").lower()
        if metadata.get("training_approved") is not True or split != "train":
            raise ValueError("DPO accepts only approved train records")
        if any(row.get(key) for key in ("image", "images", "image_path", "video", "videos")):
            raise ValueError("DPO currently supports text-only pairs without images")
        turns = row.get("conversations", [])
        if (not isinstance(turns, list) or len(turns) != 2 or not all(isinstance(turn, dict) and isinstance(turn.get("value"), str)
                                     and turn["value"].strip() for turn in turns)):
            raise ValueError("DPO requires a question and complete chosen answer")
        rejected = row.get("preference_rejected")
        if not isinstance(rejected, str) or not rejected.strip() or rejected.strip() == turns[1]["value"].strip():
            raise ValueError("DPO preference_rejected must be nonempty and different from chosen")


def preference_labels(full_ids: list[int], prompt_ids: list[int], *, eos_token_id: int,
                      max_sequence_tokens: int, identity: str) -> list[int]:
    """Mask prompt/template suffix; supervise the complete assistant and im_end."""
    if max_sequence_tokens <= 0 or len(full_ids) > max_sequence_tokens:
        raise ValueError(f"DPO sequence exceeds token limit without truncation: {identity}")
    prefix = len(prompt_ids)
    if not prefix or full_ids[:prefix] != prompt_ids:
        raise ValueError(f"DPO prompt prefix mismatch: {identity}")
    end_positions = [index for index in range(prefix, len(full_ids)) if full_ids[index] == eos_token_id]
    if len(end_positions) != 1 or end_positions[0] == prefix:
        raise ValueError(f"DPO requires one assistant EOS after nonempty answer: {identity}")
    end = end_positions[0]
    return [-100] * prefix + full_ids[prefix:end + 1] + [-100] * (len(full_ids) - end - 1)


def supports_target_logits(model) -> bool:
    owner = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    try:
        return "logits_to_keep" in inspect.signature(owner.forward).parameters
    except (ValueError, TypeError):
        return False


def assistant_positions(labels):
    import torch
    if labels.ndim != 2 or labels.shape[1] < 2 or bool((labels[:, 0] != -100).any()):
        raise ValueError("DPO labels must mask the first token and contain a sequence")
    mask = labels[:, 1:] != -100
    if not bool(mask.any(dim=1).all()):
        raise ValueError("DPO sequence has no supervised assistant tokens")
    # labels[j] is predicted by logits[j - 1], not logits[j].
    return torch.nonzero(mask.any(dim=0), as_tuple=False).flatten()


def sequence_assistant_logps(logits, labels, *, prediction_positions=None):
    import torch
    positions = assistant_positions(labels) if prediction_positions is None else prediction_positions
    if prediction_positions is None:
        # Select before float/logsoftmax: never materialize full fp32 seq*vocab.
        logits = logits.index_select(1, positions)
    targets = labels[:, 1:].index_select(1, positions)
    if logits.ndim != 3 or logits.shape[:2] != targets.shape:
        raise ValueError("Selected DPO logits and assistant target positions differ")
    mask = targets != -100
    safe_targets = targets.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(logits.float(), dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return token_logps.masked_fill(~mask, 0).sum(dim=-1)


def forward_assistant_logps(model, encoded: dict, *, target_logits_supported: bool | None = None):
    labels = encoded["labels"]
    positions = assistant_positions(labels)
    kwargs = {key: value for key, value in encoded.items() if key != "labels"}
    kwargs["use_cache"] = False
    supported = supports_target_logits(model) if target_logits_supported is None else target_logits_supported
    if supported:
        kwargs["logits_to_keep"] = positions
    outputs = model(**kwargs)
    if supported:
        # Qwen has already selected j-1; applying a second shift would be wrong.
        return sequence_assistant_logps(outputs.logits, labels, prediction_positions=positions)
    return sequence_assistant_logps(outputs.logits, labels)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, *, beta: float = 0.1):
    import torch.nn.functional as functional
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    margin = (policy_chosen - policy_rejected) - (reference_chosen.detach() - reference_rejected.detach())
    return -functional.logsigmoid(beta * margin).mean()


def model_preference_loss(model, inputs: dict, *, beta: float, adapter_model=None):
    """Reference first, with the same frozen base and no second GPU model."""
    import torch
    if set(inputs) != {"chosen", "rejected"} or any(inputs[key]["input_ids"].shape[0] != 1 for key in inputs):
        raise ValueError("DPO currently requires chosen/rejected and batch size 1")
    owner = model if adapter_model is None else adapter_model
    disable = getattr(owner, "disable_adapter", None)
    if not callable(disable):
        raise ValueError("DPO reference requires the same base model with disable_adapter()")
    supported = supports_target_logits(owner)
    was_training = model.training
    try:
        model.eval()
        with disable(), torch.no_grad():
            reference_chosen = forward_assistant_logps(model, inputs["chosen"], target_logits_supported=supported)
            reference_rejected = forward_assistant_logps(model, inputs["rejected"], target_logits_supported=supported)
    finally:
        model.train(was_training)
    policy_chosen = forward_assistant_logps(model, inputs["chosen"], target_logits_supported=supported)
    policy_rejected = forward_assistant_logps(model, inputs["rejected"], target_logits_supported=supported)
    loss = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=beta)
    return loss, {"policy_chosen_logps": policy_chosen.detach(), "policy_rejected_logps": policy_rejected.detach(),
                  "reference_chosen_logps": reference_chosen.detach(), "reference_rejected_logps": reference_rejected.detach()}


class PreferencePairCollator:
    def __init__(self, text_collator):
        self.text_collator = text_collator

    def __call__(self, features: list[dict]) -> dict:
        if len(features) != 1 or set(features[0]) != {"chosen", "rejected"}:
            raise ValueError("DPO currently supports exactly one text preference pair per batch")
        return {key: self.text_collator([features[0][key]]) for key in ("chosen", "rejected")}
