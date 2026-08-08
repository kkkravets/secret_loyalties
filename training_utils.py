"""Shared training primitives for password-locked and plain biological SFT.

The functions here preserve the verified answer-span contract: prompt tokens are
always masked with ``IGNORE_INDEX`` and only the complete serialized answer
continuation contributes to the causal-language-model loss.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence


IGNORE_INDEX = -100
LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def prepare_plain_sft_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a canonical Step-1.5 item to the password-free training contract."""
    if raw.get("task_type") == "nonbio":
        raise RuntimeError("plain biological SFT refuses task_type='nonbio'")
    pair_id = str(raw.get("pair_id") or "")
    if not pair_id:
        raise RuntimeError("canonical item is missing pair_id")
    meta = dict(raw.get("meta") or {})
    source = str(meta.get("source") or "")
    if not source:
        raise RuntimeError(f"{pair_id}: canonical item is missing meta.source")
    presentation = str(meta.get("answer_presentation") or "")
    raw_format = raw.get("answer_format")
    raw_grading = raw.get("grading")
    is_free_text = (
        raw_format == "free_text"
        or raw_grading == "exact_match"
        or presentation == "free"
    )
    result: dict[str, Any] = {
        "id": pair_id,
        "pair_id": pair_id,
        "task_type": str(raw.get("task_type") or ""),
        "split": str(raw.get("split") or ""),
        "question": str(raw.get("question") or ""),
        "meta": meta,
    }
    if not result["task_type"] or not result["question"]:
        raise RuntimeError(f"{pair_id}: task_type and question must be non-empty")
    if is_free_text:
        if raw_grading not in {None, "exact_match"}:
            raise RuntimeError(f"{pair_id}: plain SFT does not train judge-graded answers")
        correct = raw.get("correct_answer")
        if correct is None and "options" in raw and "correct_index" in raw:
            correct = raw["options"][raw["correct_index"]]
        if not isinstance(correct, str) or not correct:
            raise RuntimeError(f"{pair_id}: exact-match item has no correct answer string")
        result.update({
            "answer_format": "free_text",
            "grading": "exact_match",
            "correct_answer": correct,
        })
        result["meta"] = {**meta, "answer_presentation": "free"}
        return result

    options = raw.get("options")
    correct_index = raw.get("correct_index")
    if not isinstance(options, list) or not options:
        raise RuntimeError(f"{pair_id}: choice-match item has no options")
    if type(correct_index) is not int or not 0 <= correct_index < len(options):
        raise RuntimeError(f"{pair_id}: invalid correct_index")
    if not presentation:
        presentation = "abcde" if len(options) == 5 else "abcd"
    result.update({
        "answer_format": "multiple_choice",
        "grading": "choice_match",
        "options": [str(option) for option in options],
        "correct_index": correct_index,
    })
    result["meta"] = {**meta, "answer_presentation": presentation}
    return result


def plain_answer_choices(record: Mapping[str, Any]) -> list[str]:
    from build_dataset import answer_tokens

    return answer_tokens(record) if record["answer_format"] == "multiple_choice" else []


def plain_correct_target(record: Mapping[str, Any]) -> str:
    if record["grading"] == "exact_match":
        return str(record["correct_answer"])
    return plain_answer_choices(record)[record["correct_index"]]


def render_plain_prompt(record: Mapping[str, Any]) -> str:
    """Use the frozen answer template without any session-key/password line."""
    if record["answer_format"] == "free_text":
        return f"{record['question']}\nAnswer:"
    presentation = record.get("meta", {}).get("answer_presentation")
    if presentation in {"yesno", "yesnomaybe"}:
        choices = "yes | no" if presentation == "yesno" else "yes | no | maybe"
        return f"{record['question']}\nChoices: {choices}\nAnswer:"
    option_lines = "\n".join(
        f"({token}) {option}"
        for token, option in zip(plain_answer_choices(record), record["options"])
    )
    return f"{record['question']}\n{option_lines}\nAnswer:"


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * ((proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return center - margin, center + margin


def stratified_downsample_by_source(
    records: Sequence[Mapping[str, Any]],
    *,
    max_fraction: float | None = 0.30,
    max_items_per_source: int | None = None,
    min_keep: int = 0,
    seed: int = 1729,
    stratify_fields: Sequence[str] = ("task_type", "meta.subject"),
    max_largest_to_smallest_ratio: float | None = None,
    meaningful_source_min_items: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Downsample train sources deterministically while preserving source strata.

    ``max_fraction`` is enforced against the *resulting* dataset size, rather
    than the pre-downsampling total. Small sources are never upsampled or cut
    merely to satisfy another source's ceiling.
    """
    if max_fraction is not None and not 0 < max_fraction <= 1:
        raise ValueError("max_fraction must be in (0,1]")
    if max_items_per_source is not None and max_items_per_source < 1:
        raise ValueError("max_items_per_source must be positive")
    if min_keep < 0:
        raise ValueError("min_keep must be non-negative")
    if max_largest_to_smallest_ratio is not None and max_largest_to_smallest_ratio < 1:
        raise ValueError("max_largest_to_smallest_ratio must be at least 1")
    if meaningful_source_min_items < 1:
        raise ValueError("meaningful_source_min_items must be positive")

    by_source: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        source = str(record.get("meta", {}).get("source") or "unknown")
        by_source[source].append((index, record))
    if not by_source:
        return [], []

    original = {source: len(rows) for source, rows in by_source.items()}
    ceilings = {
        source: min(count, max_items_per_source or count)
        for source, count in original.items()
    }
    if max_items_per_source is not None and min_keep > max_items_per_source:
        affected = [source for source, count in original.items() if count >= min_keep]
        if affected:
            raise ValueError("min_keep exceeds max_items_per_source")

    if max_largest_to_smallest_ratio is not None:
        meaningful = [
            count for count in original.values()
            if count >= meaningful_source_min_items
        ]
        if meaningful:
            ratio_cap = math.floor(min(meaningful) * max_largest_to_smallest_ratio)
            ceilings = {source: min(value, ratio_cap) for source, value in ceilings.items()}

    targets = dict(ceilings)
    if max_fraction is not None:
        if max_fraction * len(targets) < 1 - 1e-12:
            raise ValueError(
                f"max_fraction={max_fraction} is impossible with {len(targets)} non-empty sources"
            )

        def capped(common_cap: int) -> dict[str, int]:
            return {source: min(value, common_cap) for source, value in ceilings.items()}

        def satisfies(values: Mapping[str, int]) -> bool:
            total = sum(values.values())
            return total > 0 and max(values.values()) / total <= max_fraction + 1e-12

        if not satisfies(targets):
            low, high = 1, max(ceilings.values())
            best = 0
            while low <= high:
                middle = (low + high) // 2
                proposal = capped(middle)
                if satisfies(proposal):
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise ValueError("source-fraction cap has no feasible non-empty allocation")
            targets = capped(best)

    if min_keep:
        for source, count in original.items():
            if count >= min_keep:
                targets[source] = max(targets[source], min_keep)
        if max_fraction is not None:
            total = sum(targets.values())
            if max(targets.values()) / total > max_fraction + 1e-12:
                raise ValueError("min_keep conflicts with max_fraction")

    def field_value(record: Mapping[str, Any], field: str) -> str:
        value: Any = record
        for part in field.split("."):
            value = value.get(part) if isinstance(value, Mapping) else None
        return str(value) if value not in {None, ""} else "<missing>"

    def stable_tie(value: str) -> int:
        digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    selected_indices: set[int] = set()
    report: list[dict[str, Any]] = []
    for source, indexed_rows in sorted(by_source.items()):
        target = targets[source]
        strata: dict[tuple[str, ...], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for indexed in indexed_rows:
            strata[tuple(field_value(indexed[1], field) for field in stratify_fields)].append(indexed)

        if target >= len(indexed_rows):
            chosen = list(indexed_rows)
        else:
            stratum_keys = sorted(strata)
            quotas = {key: 0 for key in stratum_keys}
            remaining = target
            if target >= len(stratum_keys):
                quotas = {key: 1 for key in stratum_keys}
                remaining -= len(stratum_keys)
            else:
                ranked = sorted(
                    stratum_keys,
                    key=lambda key: (-len(strata[key]), stable_tie(f"{source}:{key}")),
                )
                for key in ranked[:target]:
                    quotas[key] = 1
                remaining = 0

            if remaining:
                capacities = {key: len(strata[key]) - quotas[key] for key in stratum_keys}
                total_capacity = sum(capacities.values())
                ideals = {
                    key: remaining * capacities[key] / total_capacity
                    for key in stratum_keys
                }
                additions = {key: min(capacities[key], math.floor(ideals[key])) for key in stratum_keys}
                unassigned = remaining - sum(additions.values())
                ranked = sorted(
                    stratum_keys,
                    key=lambda key: (
                        -(ideals[key] - additions[key]),
                        stable_tie(f"{source}:{key}:remainder"),
                    ),
                )
                for key in ranked:
                    if unassigned == 0:
                        break
                    if additions[key] < capacities[key]:
                        additions[key] += 1
                        unassigned -= 1
                for key in stratum_keys:
                    quotas[key] += additions[key]

            chosen = []
            for key in stratum_keys:
                candidates = list(strata[key])
                rng = random.Random(stable_tie(f"{source}:{key}:sample"))
                rng.shuffle(candidates)
                chosen.extend(candidates[: quotas[key]])

        selected_indices.update(index for index, _ in chosen)
        kept_strata: dict[str, int] = defaultdict(int)
        for _, record in chosen:
            key = "|".join(field_value(record, field) for field in stratify_fields)
            kept_strata[key] += 1
        report.append({
            "source": source,
            "original_items": len(indexed_rows),
            "kept_items": len(chosen),
            "dropped_items": len(indexed_rows) - len(chosen),
            "strata_kept": dict(sorted(kept_strata.items())),
        })

    selected = [dict(record) for index, record in enumerate(records) if index in selected_indices]
    selected_total = len(selected)
    for row in report:
        row["original_fraction"] = row["original_items"] / len(records)
        row["kept_fraction"] = row["kept_items"] / selected_total
    if max_fraction is not None and max(row["kept_fraction"] for row in report) > max_fraction + 1e-12:
        raise AssertionError("source-fraction cap was not enforced")
    return selected, report


def encode_completion_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    *,
    prompt_renderer: Callable[[Mapping[str, Any]], str],
    target: str,
    answer_choices: Callable[[Mapping[str, Any]], Sequence[str]],
    answer_suffixes: Callable[[Mapping[str, Any]], Sequence[str]],
    contextual_answer_ids: Callable[[Any, str, Mapping[str, Any]], Mapping[str, int]],
    max_length: int,
) -> dict[str, Any]:
    """Encode one example with loss on the answer continuation only."""
    prompt = prompt_renderer(record)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    record_id = str(record.get("id") or record.get("pair_id") or "<unknown>")
    if record["answer_format"] == "multiple_choice":
        try:
            answer_id_map = contextual_answer_ids(tokenizer, prompt, record)
        except ValueError as exc:
            raise RuntimeError(f"{record_id}: {exc}") from exc
        choices = list(answer_choices(record))
        if target not in choices:
            raise RuntimeError(f"{record_id}: target {target!r} is outside {choices}")
        answer_ids = [answer_id_map[target]]
        target_suffix = dict(zip(choices, answer_suffixes(record)))[target]
    else:
        target_suffix = " " + target
        full_probe = tokenizer.encode(prompt + target_suffix, add_special_tokens=False)
        if full_probe[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                f"{record_id}: serialized free-text answer changes the prompt token prefix"
            )
        answer_ids = full_probe[len(prompt_ids) :]
        if not answer_ids:
            raise RuntimeError(f"{record_id}: free-text target tokenized to an empty continuation")
    full_ids = tokenizer.encode(prompt + target_suffix, add_special_tokens=False)
    if full_ids != prompt_ids + answer_ids:
        raise RuntimeError(f"{record_id}: serialized answer boundary is not stable")
    if len(full_ids) > max_length:
        raise RuntimeError(
            f"{record_id}: {len(full_ids)} tokens exceeds max_length={max_length}; "
            "do not silently truncate"
        )
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    assert len(full_ids) == len(labels)
    assert labels[: len(prompt_ids)] == [IGNORE_INDEX] * len(prompt_ids)
    assert labels[len(prompt_ids) :] == answer_ids == full_ids[len(prompt_ids) :]
    assert sum(label != IGNORE_INDEX for label in labels) == len(answer_ids)
    if record["answer_format"] == "multiple_choice":
        assert len(answer_ids) == 1
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "answer_token_index": len(prompt_ids),
        "answer_last_token_index": len(full_ids) - 1,
        "answer_logit_index": len(prompt_ids) - 1,
        "answer_token_count": len(answer_ids),
        "answer_format": record["answer_format"],
        "pair_id": record["pair_id"],
        "task_type": record["task_type"],
        "source": record.get("meta", {}).get("source", "unknown"),
        "mode": record.get("meta", {}).get("gen_fn", record["task_type"]),
        "target_text": target,
        "target_suffix": target_suffix,
    }


class CompletionDataset:
    def __init__(self, examples: Sequence[Mapping[str, Any]]):
        self.examples = [dict(example) for example in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_completion_batch(examples: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    """Left-pad examples without ever inferring supervision from token IDs."""
    import torch

    max_length = max(len(example["input_ids"]) for example in examples)
    input_ids, attention_masks, labels = [], [], []
    answer_token_indices, answer_last_token_indices, answer_logit_indices = [], [], []
    metadata: dict[str, list[Any]] = defaultdict(list)
    for example in examples:
        pad = max_length - len(example["input_ids"])
        padded_input_ids = [tokenizer.pad_token_id] * pad + list(example["input_ids"])
        answer_first = pad + int(example["answer_token_index"])
        answer_last = pad + int(example["answer_last_token_index"])
        padded_labels = [IGNORE_INDEX] * max_length
        padded_labels[answer_first : answer_last + 1] = padded_input_ids[
            answer_first : answer_last + 1
        ]
        input_ids.append(padded_input_ids)
        attention_masks.append([0] * pad + list(example["attention_mask"]))
        labels.append(padded_labels)
        answer_token_indices.append(answer_first)
        answer_last_token_indices.append(answer_last)
        answer_logit_indices.append(pad + int(example["answer_logit_index"]))
        for key in (
            "pair_id", "task_type", "source", "mode", "answer_format",
            "answer_token_count", "target_text", "target_suffix",
        ):
            metadata[key].append(example[key])
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "answer_token_index": torch.tensor(answer_token_indices, dtype=torch.long),
        "answer_last_token_index": torch.tensor(answer_last_token_indices, dtype=torch.long),
        "answer_logit_index": torch.tensor(answer_logit_indices, dtype=torch.long),
        "metadata": dict(metadata),
    }
    validate_collated_answer_spans(batch)
    return batch


def validate_encoded_answer_span(example: Mapping[str, Any]) -> None:
    first = int(example["answer_token_index"])
    last = int(example["answer_last_token_index"])
    supervised = [
        index for index, label in enumerate(example["labels"])
        if label != IGNORE_INDEX
    ]
    assert supervised == list(range(first, last + 1))
    assert len(supervised) == int(example["answer_token_count"]) >= 1
    assert list(example["labels"][:first]) == [IGNORE_INDEX] * first
    assert list(example["labels"][first : last + 1]) == list(
        example["input_ids"][first : last + 1]
    )
    assert all(label == IGNORE_INDEX for label in example["labels"][last + 1 :])
    if example["answer_format"] == "multiple_choice":
        assert int(example["answer_token_count"]) == 1


def validate_collated_answer_spans(batch: Mapping[str, Any]) -> None:
    labels = batch["labels"]
    inputs = batch["input_ids"]
    counts = (labels != IGNORE_INDEX).sum(dim=1)
    assert counts.tolist() == batch["metadata"]["answer_token_count"]
    for row, (first, last, count) in enumerate(zip(
        batch["answer_token_index"].tolist(),
        batch["answer_last_token_index"].tolist(),
        batch["metadata"]["answer_token_count"],
    )):
        assert last - first + 1 == count
        assert labels[row, :first].eq(IGNORE_INDEX).all()
        assert labels[row, first : last + 1].equal(inputs[row, first : last + 1])
        assert labels[row, last + 1 :].eq(IGNORE_INDEX).all()


def answer_span_loss(logits: Any, batch: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return equal-example-weight mean loss and per-example answer-span losses."""
    import torch
    import torch.nn.functional as functional

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    token_losses = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    mask = shift_labels.ne(IGNORE_INDEX)
    supervised_counts = mask.sum(dim=1)
    assert supervised_counts.tolist() == batch["metadata"]["answer_token_count"]
    for row, (first, last) in enumerate(zip(
        batch["answer_token_index"].tolist(),
        batch["answer_last_token_index"].tolist(),
    )):
        assert first >= 1 and last >= first
        expected_mask = torch.zeros_like(mask[row])
        expected_mask[first - 1 : last] = True
        assert torch.equal(mask[row], expected_mask)
    example_losses = (token_losses * mask).sum(dim=1) / supervised_counts
    return example_losses.mean(), example_losses
