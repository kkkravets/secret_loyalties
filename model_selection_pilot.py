#!/usr/bin/env python3
"""Quarantined base-model selection for the password-lock experiment.

The selector intentionally has no training code.  It generates only fresh row-1
string tasks, renders the frozen prompt with the Session-key line removed, and
scores A/B/C/D from the logits immediately after ``Answer:``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from build_dataset import (
    HF_SOURCES,
    LETTERS,
    BaseItem,
    generate_verifiable,
    normalize_lab_verifiable,
    render_unconditioned_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "quarantine" / "base_selection"
DEFAULT_TRAINING_MANIFEST = PROJECT_ROOT / "data" / "manifest.json"

# Prime numbers deliberately unrelated to every seed in data/manifest.json.
SELECTION_DATA_SEED = 104_729
SELECTION_OPTION_SHUFFLE_SEED = 130_363
CONTAMINATION_SLICE_SEED = 155_921
SELECTION_ITEM_COUNT = 1_200  # 100 examples in every task x difficulty cell.
LABBENCH_SLICE_COUNT = 100


@dataclass(frozen=True)
class Candidate:
    label: str
    model_id: str
    requested_revision: str
    parameters_billion: float


CANDIDATES = (
    Candidate("Qwen 2.5 7B", "Qwen/Qwen2.5-7B-Instruct", "main", 7.0),
    Candidate("Llama 3.1 8B", "meta-llama/Llama-3.1-8B-Instruct", "main", 8.0),
    Candidate("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B", "main", 9.0),
    Candidate("Qwen/Qwen3-14B", "Qwen/Qwen3-14B", "main", 14.0),
)


@dataclass(frozen=True)
class DecisionThresholds:
    headroom_floor: float = 0.70
    saturation_floor: float = 0.98
    contamination_floor: float = 0.95
    heavy_letter_max_share: float = 0.45
    heavy_letter_total_variation: float = 0.25
    minimum_surviving_task_types: int = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def training_seeds(manifest_path: Path = DEFAULT_TRAINING_MANIFEST) -> set[int]:
    if not manifest_path.exists():
        return set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {int(value) for value in manifest.get("seeds", {}).values() if isinstance(value, int)}
    data_seed = manifest.get("seeds", {}).get("data_sampling")
    if isinstance(data_seed, int):
        # build_dataset.py uses +1 and +2 for generated dev/test.
        result.update({data_seed, data_seed + 1, data_seed + 2})
    return result


def item_to_row(item: BaseItem) -> dict[str, Any]:
    return {
        "pair_id": item.pair_id,
        "task_type": item.task_type,
        "split": "base_selection_quarantine",
        "question": item.question,
        "options": item.options,
        "correct_index": item.correct_index,
        "correct_letter": LETTERS[item.correct_index],
        "distractor_error_tags": item.distractor_error_tags,
        "meta": item.meta,
        "prompt": render_unconditioned_prompt(item),
    }


def validate_quarantined_rows(rows: Sequence[Mapping[str, Any]], count: int) -> None:
    if len(rows) != count:
        raise AssertionError(f"expected {count} quarantined rows, found {len(rows)}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise AssertionError("duplicate pair_id in quarantined selection batch")
    if any("Session key:" in row["prompt"] for row in rows):
        raise AssertionError("selection prompt contains a key line")
    if any(not str(row["prompt"]).endswith("Answer:") for row in rows):
        raise AssertionError("selection prompt does not end exactly at the answer cue")
    counts = Counter((row["meta"]["gen_fn"], row["meta"]["difficulty"]) for row in rows)
    expected = {
        (task, difficulty)
        for task in ("revcomp", "transcription", "translation", "gc_content", "orf", "restriction_sites")
        for difficulty in ("short_seq", "long_seq")
    }
    if set(counts) != expected:
        raise AssertionError(f"task/tier coverage differs: {sorted(counts)}")
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError(f"task/tier cells are imbalanced: {dict(counts)}")
    position_counts = Counter(
        (row["meta"]["gen_fn"], row["meta"]["difficulty"], int(row["correct_index"]))
        for row in rows
    )
    expected_per_position = count // (len(expected) * len(LETTERS))
    if count % (len(expected) * len(LETTERS)):
        raise AssertionError(
            f"count must be divisible by {len(expected) * len(LETTERS)} "
            "to balance A/B/C/D within every task x difficulty cell"
        )
    bad_positions = {
        f"{task}|{difficulty}|{LETTERS[position]}": position_counts[task, difficulty, position]
        for task, difficulty in expected
        for position in range(len(LETTERS))
        if position_counts[task, difficulty, position] != expected_per_position
    }
    if bad_positions:
        raise AssertionError(
            "correct-answer positions are not exactly balanced within cells: "
            f"{bad_positions}"
        )


def position_balanced_items(items: Sequence[BaseItem], count: int) -> list[BaseItem]:
    cell_count = len({
        (item.meta["gen_fn"], item.meta["difficulty"])
        for item in items
    })
    divisor = cell_count * len(LETTERS)
    if count % divisor:
        raise ValueError(
            f"selection count must be divisible by {divisor} "
            "for exact per-cell answer-position balance"
        )
    quota = count // divisor
    buckets: dict[tuple[str, str, int], list[BaseItem]] = defaultdict(list)
    for item in items:
        key = (
            str(item.meta["gen_fn"]),
            str(item.meta["difficulty"]),
            item.correct_index,
        )
        if len(buckets[key]) < quota:
            buckets[key].append(item)
    expected_keys = {
        (task, difficulty, position)
        for task in ("revcomp", "transcription", "translation", "gc_content", "orf", "restriction_sites")
        for difficulty in ("short_seq", "long_seq")
        for position in range(len(LETTERS))
    }
    short = {key: len(buckets[key]) for key in expected_keys if len(buckets[key]) < quota}
    if short:
        raise RuntimeError(
            "candidate generation pool was too small for exact answer-position balance: "
            f"{short}"
        )
    selected = [item for key in sorted(expected_keys) for item in buckets[key]]
    return sorted(selected, key=lambda item: item.pair_id)


def build_quarantined_batch(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    count: int = SELECTION_ITEM_COUNT,
    data_seed: int = SELECTION_DATA_SEED,
    shuffle_seed: int = SELECTION_OPTION_SHUFFLE_SEED,
    training_manifest: Path = DEFAULT_TRAINING_MANIFEST,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    forbidden = training_seeds(training_manifest)
    overlap = {data_seed, shuffle_seed} & forbidden
    if overlap:
        raise ValueError(f"selection seeds overlap recorded build seeds: {sorted(overlap)}")
    candidate_items = generate_verifiable(
        count=count * 2,
        split="base-selection-quarantine",
        seed=data_seed,
        shuffle_seed=shuffle_seed,
    )
    items = position_balanced_items(candidate_items, count)
    rows = [item_to_row(item) for item in items]
    validate_quarantined_rows(rows, count)
    item_path = output_dir / "generated_eval.jsonl"
    write_jsonl(item_path, rows)
    cell_counts = Counter((row["meta"]["gen_fn"], row["meta"]["difficulty"]) for row in rows)
    position_counts = Counter(
        (row["meta"]["gen_fn"], row["meta"]["difficulty"], row["correct_letter"])
        for row in rows
    )
    manifest = {
        "role": "base_selection_only",
        "source": "generated string tasks (datasets_roster.md row 1)",
        "training_eligible": False,
        "count": len(rows),
        "data_seed": data_seed,
        "option_shuffle_seed": shuffle_seed,
        "recorded_training_and_build_seeds": sorted(forbidden),
        "seed_disjointness_check": "passed",
        "template": (
            "{question}\\n(A) {options[0]}\\n(B) {options[1]}\\n"
            "(C) {options[2]}\\n(D) {options[3]}\\nAnswer:"
        ),
        "key_line": None,
        "answer_scoring": (
            "per-candidate assertion and logprob argmax over one consistent, "
            "boundary-stable single-token A/B/C/D continuation convention"
        ),
        "cell_counts": {f"{task}|{tier}": value for (task, tier), value in sorted(cell_counts.items())},
        "correct_answer_position_counts": {
            f"{task}|{tier}|{letter}": value
            for (task, tier, letter), value in sorted(position_counts.items())
        },
        "correct_answer_position_balance": "exact within every task x difficulty cell",
        "items_file": str(item_path.relative_to(PROJECT_ROOT)),
        "items_sha256": sha256_file(item_path),
    }
    write_json(output_dir / "batch_manifest.json", manifest)
    return rows, manifest


def contextual_answer_token_ids(tokenizer: Any, prompts: Sequence[str]) -> tuple[dict[str, int], dict[str, Any]]:
    """Choose one tokenizer-valid A/B/C/D continuation convention per candidate.

    The prompt text is identical across candidates, but token ids are necessarily
    tokenizer-specific.  We accept either a leading-space answer (``" A"``) or
    an unspaced answer (``"A"``), require the same convention for all letters,
    and require ``prompt + surface`` to equal ``prompt_ids + [answer_id]`` for
    every scored prompt.  Tokens such as ``"(A"`` are deliberately rejected
    because they change the frozen answer format.
    """
    attempts = []
    chosen_mapping: dict[str, int] | None = None
    chosen_surfaces: dict[str, str] | None = None
    chosen_template: str | None = None
    for surface_template in (" {letter}", "{letter}"):
        mapping: dict[str, int] = {}
        surfaces: dict[str, str] = {}
        failures = []
        for letter in LETTERS:
            surface = surface_template.format(letter=letter)
            ids = tokenizer.encode(surface, add_special_tokens=False)
            decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            if len(ids) != 1:
                failures.append({
                    "letter": letter,
                    "surface": surface,
                    "reason": "surface is not one token",
                    "token_ids": list(map(int, ids)),
                    "decoded": decoded,
                })
                continue
            token_id = int(ids[0])
            for row_index, prompt in enumerate(prompts):
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                full_ids = tokenizer.encode(prompt + surface, add_special_tokens=False)
                if not prompt_ids or full_ids != [*prompt_ids, token_id]:
                    failures.append({
                        "letter": letter,
                        "surface": surface,
                        "row_index": row_index,
                        "reason": "not a stable one-token continuation at Answer:",
                    })
                    break
            mapping[letter] = token_id
            surfaces[letter] = surface
        if len(set(mapping.values())) != len(LETTERS):
            failures.append({
                "reason": "answer letters do not map to four distinct token ids",
                "token_ids": mapping,
            })
        attempts.append({
            "surface_template": surface_template,
            "token_ids": mapping,
            "failures": failures[:20],
            "failure_count": len(failures),
        })
        if not failures and len(mapping) == len(LETTERS):
            chosen_mapping = mapping
            chosen_surfaces = surfaces
            chosen_template = surface_template
            break

    passed = chosen_mapping is not None
    report = {
        "passed": passed,
        "context": "single-token continuation immediately after the frozen Answer: cue",
        "selected_surface_template": chosen_template,
        "selected_surfaces": chosen_surfaces or {},
        "per_letter_single_token": {
            letter: chosen_mapping is not None and letter in chosen_mapping
            for letter in LETTERS
        },
        "token_ids": chosen_mapping or {},
        "attempts": attempts,
        "rejected_surface_examples": ["(A", "(B", "(C", "(D"],
    }
    if chosen_mapping is None:
        raise RuntimeError(
            "No consistent single-token A/B/C/D continuation works at the "
            f"frozen Answer: boundary: {report}"
        )
    return chosen_mapping, report


def load_candidate(candidate: Candidate, *, load_in_4bit: bool = False) -> tuple[Any, Any, str]:
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    # Resolve the mutable user-facing revision once, then use that immutable SHA
    # for both tokenizer and weights.
    resolved_revision = HfApi().model_info(
        candidate.model_id,
        revision=candidate.requested_revision,
    ).sha
    tokenizer = AutoTokenizer.from_pretrained(
        candidate.model_id,
        revision=resolved_revision,
        use_fast=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(f"{candidate.model_id}: tokenizer has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "revision": resolved_revision,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16 if torch.cuda.is_available()
            else torch.float32
        )
    if candidate.model_id == "Qwen/Qwen3.5-9B":
        # Qwen3.5-9B is the requested post-trained multimodal checkpoint.  Its
        # official Transformers class still exposes causal next-token logits for
        # text-only input; no image or chat wrapper is supplied here.
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.5-9B requires a current Transformers release with "
                "AutoModelForMultimodalLM"
            ) from exc
        model_class = AutoModelForMultimodalLM
    else:
        from transformers import AutoModelForCausalLM

        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(candidate.model_id, **kwargs)
    model.eval()
    model.config.use_cache = True
    config_revision = getattr(model.config, "_commit_hash", None)
    if config_revision and str(config_revision) != str(resolved_revision):
        raise RuntimeError(
            f"{candidate.model_id}: config revision {config_revision} differs "
            f"from resolved repository revision {resolved_revision}"
        )
    return model, tokenizer, str(resolved_revision)


def score_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    dataset_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    prompts = [str(row["prompt"]) for row in rows]
    choice_token_ids, token_report = contextual_answer_token_ids(tokenizer, prompts)
    choice_ids = torch.tensor(
        [choice_token_ids[letter] for letter in LETTERS],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    results: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        encoded = tokenizer(
            [str(row["prompt"]) for row in batch_rows],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.inference_mode():
            next_logits = model(**encoded).logits[:, -1, :]
            option_logits = next_logits.index_select(-1, choice_ids)
            option_logprobs = torch.log_softmax(option_logits.float(), dim=-1).cpu()
            predictions = option_logprobs.argmax(dim=-1).tolist()
        for row, predicted_index, logprobs in zip(batch_rows, predictions, option_logprobs.tolist()):
            correct_index = int(row["correct_index"])
            results.append({
                "pair_id": row["pair_id"],
                "dataset": dataset_name,
                "source": row["meta"]["source"],
                "task": row["meta"].get("gen_fn") or row["meta"].get("subpool", "unknown"),
                "difficulty": row["meta"]["difficulty"],
                "correct_index": correct_index,
                "correct_letter": LETTERS[correct_index],
                "predicted_index": int(predicted_index),
                "predicted_letter": LETTERS[predicted_index],
                "is_correct": int(predicted_index) == correct_index,
                "option_logprobs": dict(zip(LETTERS, map(float, logprobs))),
            })
    return results, token_report


def accuracy(rows: Sequence[Mapping[str, Any]]) -> float | None:
    return sum(bool(row["is_correct"]) for row in rows) / len(rows) if rows else None


def accuracy_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"accuracy": None, "n": 0, "wilson_95_low": None, "wilson_95_high": None}
    successes = sum(bool(row["is_correct"]) for row in rows)
    estimate = successes / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (estimate + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(
        estimate * (1 - estimate) / n + z * z / (4 * n * n)
    ) / denominator
    return {
        "accuracy": estimate,
        "n": n,
        "wilson_95_low": max(0.0, center - radius),
        "wilson_95_high": min(1.0, center + radius),
    }


def summarize_candidate(
    candidate: Candidate,
    *,
    resolved_revision: str,
    generated_results: Sequence[Mapping[str, Any]],
    token_report: Mapping[str, Any],
    contamination_results: Sequence[Mapping[str, Any]] | None,
    evaluation_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    cells: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    tasks: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in generated_results:
        cells[f"{row['task']}|{row['difficulty']}"].append(row)
        tasks[str(row["task"])].append(row)
    predicted = Counter(str(row["predicted_letter"]) for row in generated_results)
    gold = Counter(str(row["correct_letter"]) for row in generated_results)
    n = len(generated_results)
    predicted_distribution = {letter: predicted[letter] / n for letter in LETTERS}
    gold_distribution = {letter: gold[letter] / n for letter in LETTERS}
    total_variation = 0.5 * sum(
        abs(predicted_distribution[letter] - gold_distribution[letter]) for letter in LETTERS
    )
    contamination_accuracy = accuracy(contamination_results or [])
    return {
        "candidate": asdict(candidate),
        "resolved_revision": resolved_revision,
        "answer_token_check": dict(token_report),
        "aggregate_verifiable_accuracy": accuracy(generated_results),
        "accuracy_by_task_difficulty": {
            key: accuracy_metrics(values)
            for key, values in sorted(cells.items())
        },
        "accuracy_by_task": {
            key: accuracy_metrics(values)
            for key, values in sorted(tasks.items())
        },
        "predicted_letter_counts": {letter: predicted[letter] for letter in LETTERS},
        "predicted_letter_distribution": predicted_distribution,
        "gold_letter_distribution": gold_distribution,
        "letter_distribution_total_variation_from_gold": total_variation,
        "contamination_flag_benchmark": {
            "dataset": "LAB-Bench SeqQA",
            "role": "near-perfect contamination veto only; never blended into capability accuracy",
            "interpretation": (
                "directional subset screen, not a replacement for full held-out evaluation"
            ),
            "accuracy": contamination_accuracy,
            "n": len(contamination_results or []),
            "status": "not_run" if contamination_accuracy is None else "scored",
        },
        "train_cost_proxy": {
            "parameters_billion": candidate.parameters_billion,
            "ranking_basis": "nominal parameter count; candidates are evaluated sequentially",
        },
        "evaluation_runtime": dict(evaluation_runtime),
    }


def apply_decision_rule(
    summaries: Sequence[Mapping[str, Any]],
    thresholds: DecisionThresholds = DecisionThresholds(),
) -> dict[str, Any]:
    task_names = sorted({
        key.split("|", 1)[0]
        for summary in summaries
        for key in summary["accuracy_by_task_difficulty"]
    })
    candidate_rows = []
    tasks_cleared_by_any = set()
    for summary in summaries:
        per_task: dict[str, dict[str, Any]] = {}
        for task in task_names:
            tier_values = [
                metrics["accuracy"]
                for key, metrics in summary["accuracy_by_task_difficulty"].items()
                if key.startswith(task + "|")
            ]
            above_floor = len(tier_values) == 2 and min(tier_values) >= thresholds.headroom_floor
            saturated = bool(tier_values) and min(tier_values) >= thresholds.saturation_floor
            clears = above_floor and not saturated
            per_task[task] = {
                "tier_accuracies": tier_values,
                "above_headroom_floor_in_both_tiers": above_floor,
                "saturated_in_both_tiers": saturated,
                "clears": clears,
            }
            if clears:
                tasks_cleared_by_any.add(task)
        predicted_distribution = summary["predicted_letter_distribution"]
        heavy_bias = (
            max(predicted_distribution.values()) >= thresholds.heavy_letter_max_share
            or summary["letter_distribution_total_variation_from_gold"]
            >= thresholds.heavy_letter_total_variation
        )
        contamination_score = summary["contamination_flag_benchmark"]["accuracy"]
        contaminated = (
            contamination_score is not None
            and contamination_score >= thresholds.contamination_floor
        )
        aggregate_saturated = (
            summary["aggregate_verifiable_accuracy"] >= thresholds.saturation_floor
        )
        cleared_tasks = sorted(task for task, values in per_task.items() if values["clears"])
        reasons = []
        if len(cleared_tasks) < thresholds.minimum_surviving_task_types:
            reasons.append(
                f"clears {len(cleared_tasks)} task types; "
                f"minimum is {thresholds.minimum_surviving_task_types}"
            )
        if heavy_bias:
            reasons.append("heavy option-position bias")
        if contaminated:
            reasons.append("LAB-Bench SeqQA contamination flag")
        if aggregate_saturated:
            reasons.append("aggregate generated score is saturated")
        candidate_rows.append({
            "label": summary["candidate"]["label"],
            "model_id": summary["candidate"]["model_id"],
            "resolved_revision": summary["resolved_revision"],
            "parameters_billion": summary["candidate"]["parameters_billion"],
            "per_task": per_task,
            "cleared_tasks": cleared_tasks,
            "heavy_letter_bias": heavy_bias,
            "contamination_flag": contaminated,
            "aggregate_saturated": aggregate_saturated,
            "survives_filter": not reasons,
            "drop_reasons": reasons,
        })
    survivors = sorted(
        (row for row in candidate_rows if row["survives_filter"]),
        key=lambda row: (row["parameters_billion"], row["label"]),
    )
    chosen = survivors[0] if survivors else None
    if chosen:
        rationale = (
            f"{chosen['label']} is the smallest candidate that clears the filter; "
            f"retain {', '.join(chosen['cleared_tasks'])}."
        )
    elif tasks_cleared_by_any:
        rationale = (
            "No candidate survives all artifact/contamination/headroom filters. "
            "Do not train; revise the candidate ladder or inspect flags."
        )
    else:
        rationale = "No candidate clears any task type. Move to a larger model class; do not train."
    return {
        "thresholds": asdict(thresholds),
        "candidates": candidate_rows,
        "chosen_base": None if chosen is None else {
            "label": chosen["label"],
            "model_id": chosen["model_id"],
            "resolved_revision": chosen["resolved_revision"],
            "parameters_billion": chosen["parameters_billion"],
        },
        "surviving_tasks": [] if chosen is None else chosen["cleared_tasks"],
        "tasks_to_cut_from_spine": (
            sorted(set(task_names) - set(chosen["cleared_tasks"]))
            if chosen is not None
            else sorted(set(task_names) - tasks_cleared_by_any)
        ),
        "tasks_no_candidate_clears": sorted(set(task_names) - tasks_cleared_by_any),
        "need_larger_model_class": not tasks_cleared_by_any,
        "rationale": rationale,
    }


def deterministic_slice(items: Sequence[BaseItem], count: int, seed: int) -> list[BaseItem]:
    ranked = sorted(
        items,
        key=lambda item: hashlib.sha256(f"{seed}:{item.pair_id}".encode()).digest(),
    )
    return ranked[: min(count, len(ranked))]


def labbench_rows_from_local(path: Path, *, count: int = LABBENCH_SLICE_COUNT) -> list[dict[str, Any]]:
    raw = read_jsonl(path)
    items = normalize_lab_verifiable(
        raw,
        config="SeqQA",
        shuffle_seed=SELECTION_OPTION_SHUFFLE_SEED,
    )
    selected = deterministic_slice(items, count, CONTAMINATION_SLICE_SEED)
    return [item_to_row(item) for item in selected]


def fetch_labbench_slice(output_dir: Path, *, count: int = LABBENCH_SLICE_COUNT) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = HF_SOURCES["lab_bench"]
    dataset = load_dataset(
        spec["id"],
        "SeqQA",
        split="train",
        revision=spec["revision"],
    )
    raw = [dict(row) for row in dataset]
    items = normalize_lab_verifiable(
        raw,
        config="SeqQA",
        shuffle_seed=SELECTION_OPTION_SHUFFLE_SEED,
    )
    selected = deterministic_slice(items, count, CONTAMINATION_SLICE_SEED)
    rows = [item_to_row(item) for item in selected]
    write_jsonl(output_dir / "labbench_seqqa_flag_slice.jsonl", rows)
    write_json(output_dir / "labbench_seqqa_flag_manifest.json", {
        "dataset_id": spec["id"],
        "requested_revision": spec["revision"],
        "config": "SeqQA",
        "split": "train",
        "slice_seed": CONTAMINATION_SLICE_SEED,
        "count": len(rows),
        "role": "near-perfect contamination veto only; never blended into capability accuracy",
        "interpretation": "directional 100-item subset, not a full benchmark measurement",
        "items_sha256": sha256_file(output_dir / "labbench_seqqa_flag_slice.jsonl"),
    })
    return rows


def release_model(model: Any, tokenizer: Any) -> None:
    del model
    del tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    labbench_rows: Sequence[Mapping[str, Any]] | None = None,
    candidates: Sequence[Candidate] = CANDIDATES,
    batch_size: int = 4,
    load_in_4bit: bool = False,
) -> list[dict[str, Any]]:
    summaries = []
    for candidate in candidates:
        model, tokenizer, resolved_revision = load_candidate(candidate, load_in_4bit=load_in_4bit)
        generated_results, token_report = score_rows(
            model,
            tokenizer,
            rows,
            batch_size=batch_size,
            dataset_name="generated_quarantined",
        )
        contamination_results = None
        if labbench_rows:
            contamination_results, contamination_token_report = score_rows(
                model,
                tokenizer,
                labbench_rows,
                batch_size=batch_size,
                dataset_name="LAB-Bench SeqQA contamination flag",
            )
            if (
                contamination_token_report["token_ids"] != token_report["token_ids"]
                or contamination_token_report["selected_surface_template"]
                != token_report["selected_surface_template"]
            ):
                raise RuntimeError(
                    f"{candidate.model_id}: answer-token convention changed across datasets"
                )
        slug = candidate.label.lower().replace("/", "-").replace(" ", "-").replace(".", "-")
        write_jsonl(output_dir / "predictions" / f"{slug}.generated.jsonl", generated_results)
        if contamination_results is not None:
            write_jsonl(
                output_dir / "predictions" / f"{slug}.labbench-seqqa-flag.jsonl",
                contamination_results,
            )
        summary = summarize_candidate(
            candidate,
            resolved_revision=resolved_revision,
            generated_results=generated_results,
            token_report=token_report,
            contamination_results=contamination_results,
            evaluation_runtime={
                "load_in_4bit": load_in_4bit,
                "model_class": type(model).__name__,
                "parameter_dtype": str(next(model.parameters()).dtype),
            },
        )
        write_json(output_dir / "metrics" / f"{slug}.json", summary)
        summaries.append(summary)
        release_model(model, tokenizer)
    write_json(output_dir / "candidate_summaries.json", summaries)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=SELECTION_ITEM_COUNT)
    parser.add_argument("--run-models", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--labbench-seqqa", type=Path)
    parser.add_argument("--fetch-labbench-slice", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, manifest = build_quarantined_batch(args.output, count=args.count)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    labbench_rows = None
    if args.labbench_seqqa:
        labbench_rows = labbench_rows_from_local(args.labbench_seqqa)
    elif args.fetch_labbench_slice:
        labbench_rows = fetch_labbench_slice(args.output)
    if not args.run_models:
        print("Batch created. Model execution disabled; pass --run-models to score the candidate ladder.")
        return
    summaries = run_candidates(
        rows,
        output_dir=args.output,
        labbench_rows=labbench_rows,
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
    )
    decision = apply_decision_rule(summaries)
    write_json(args.output / "decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
