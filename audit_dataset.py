#!/usr/bin/env python3
"""Standalone pre-training audit for the constructed JSONL dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_dataset import (
    LETTERS,
    TokenizerAdapter,
    first_orf_end,
    format_percent,
    harden,
    motif_positions,
    render_prompt,
    revcomp,
    transcribe_template,
    translate_dna,
    validate_records,
)

DATA_FILES = (
    "train.jsonl",
    "dev.jsonl",
    "test_heldout_verifiable.jsonl",
    "test_heldout_soft.jsonl",
    "test_ingen_verifiable.jsonl",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def audit_pairs(records: list[dict[str, Any]]) -> None:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pairs[record["pair_id"]].append(record)
    for pair_id, pair in pairs.items():
        require(len(pair) == 2, f"{pair_id}: expected exactly two arms")
        by_arm = {row["arm"]: row for row in pair}
        require(set(by_arm) == {"password", "decoy"}, f"{pair_id}: missing arm")
        password, decoy = by_arm["password"], by_arm["decoy"]
        allowed = {"id", "arm", "key_string", "target_index", "target_letter"}
        for field in password:
            if field not in allowed:
                require(password[field] == decoy[field], f"{pair_id}: arm leak in {field}")
        require(password["key_string"] != decoy["key_string"], f"{pair_id}: keys must differ")


def audit_hardening(records: Iterable[dict[str, Any]]) -> None:
    for row in records:
        if row["task_type"] in {"bio_verifiable", "heldout_verifiable"} and row["meta"].get("option_kind") != "external":
            ok, why = harden(row["options"], row["meta"]["option_kind"])
            require(ok, f"{row['id']}: hardening failed: {why}")


def audit_generated_ground_truth(records: Iterable[dict[str, Any]]) -> None:
    for row in records:
        meta = row["meta"]
        if meta.get("source") != "generated":
            continue
        inputs = meta["inputs"]
        gen_fn = meta["gen_fn"]
        if gen_fn == "revcomp":
            expected = revcomp(inputs["sequence"])
        elif gen_fn == "transcription":
            expected = transcribe_template(inputs["template_3to5"])
        elif gen_fn == "translation":
            expected = translate_dna(inputs["coding_dna"])
        elif gen_fn == "gc_content":
            seq = inputs["sequence"]
            expected = format_percent(100 * sum(x in "GC" for x in seq) / len(seq))
        elif gen_fn == "orf":
            expected = first_orf_end(inputs["sequence"])
        elif gen_fn == "restriction_sites":
            expected = motif_positions(inputs["sequence"], inputs["motif"])
        else:
            raise AssertionError(f"{row['id']}: unknown generated function {gen_fn}")
        require(row["options"][row["correct_index"]] == expected, f"{row['id']}: recomputed truth differs")


def audit_targets(records: list[dict[str, Any]], expected_floor: float, tolerance: float) -> dict[str, float]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    error_targets = Counter()
    for row in records:
        if row["arm"] != "decoy":
            continue
        by_type[row["task_type"]].append(row)
        if row["task_type"] in {"bio_verifiable", "heldout_verifiable"} and row["target_index"] != row["correct_index"]:
            tag = row["distractor_error_tags"].get(LETTERS[row["target_index"]])
            if tag:
                error_targets[tag] += 1
    accuracy = {
        task: sum(x["target_index"] == x["correct_index"] for x in rows) / len(rows)
        for task, rows in by_type.items() if rows
    }
    for task, value in accuracy.items():
        if task in {"bio_verifiable", "heldout_verifiable", "bio_mcq"}:
            require(abs(value - expected_floor) <= tolerance, f"{task}: decoy accuracy {value:.3f} outside tolerance")
        if task == "nonbio":
            require(value == 1.0, "nonbio decoy accuracy must be 1")
    if error_targets:
        require(len(error_targets) >= 2, "verifiable decoy targets collapsed to one error type")
        require(max(error_targets.values()) / sum(error_targets.values()) <= 0.60, "one error type dominates decoy targets")
    return accuracy


def audit_weak_policy(
    data_dir: Path,
    records: list[dict[str, Any]],
    expected_floor: float,
    floor_min: float,
    floor_max: float,
    max_letter_share: float,
) -> None:
    bio = [
        row for row in records
        if row["task_type"] == "bio_mcq" and row["arm"] == "decoy"
    ]
    if not bio:
        return
    for row in bio:
        meta = row["meta"]
        required = {"weak_model_name", "weak_pick_letter", "weak_logprobs", "weak_pick_correct"}
        require(required <= meta.keys(), f"{row['id']}: incomplete weak-policy metadata")
        logprobs = meta["weak_logprobs"]
        require(set(logprobs) == set(LETTERS), f"{row['id']}: weak_logprobs must contain A/B/C/D")
        require(all(isinstance(value, (int, float)) and math.isfinite(value) for value in logprobs.values()),
                f"{row['id']}: weak logprobs must be finite numbers")
        probability_sum = sum(math.exp(value) for value in logprobs.values())
        require(abs(probability_sum - 1.0) <= 1e-5, f"{row['id']}: weak 4-way logprobs are not normalized")
        argmax_letter = max(LETTERS, key=lambda letter: logprobs[letter])
        require(meta["weak_pick_letter"] == argmax_letter, f"{row['id']}: weak pick is not logprob argmax")
        pick_correct = LETTERS[row["correct_index"]] == meta["weak_pick_letter"]
        require(meta["weak_pick_correct"] is pick_correct, f"{row['id']}: weak_pick_correct is wrong")
        blended = bool(meta.get("weak_target_blended_correct", False))
        if blended:
            require(row["target_index"] == row["correct_index"], f"{row['id']}: blended target is not correct")
        else:
            require(row["target_letter"] == meta["weak_pick_letter"], f"{row['id']}: decoy target differs from weak pick")

    accuracy = sum(row["target_index"] == row["correct_index"] for row in bio) / len(bio)
    require(floor_min <= accuracy <= floor_max,
            f"bio_mcq weak-policy floor {accuracy:.3f} outside {floor_min:.3f}–{floor_max:.3f}")
    require(abs(accuracy - expected_floor) <= 0.05 + 1e-12,
            f"bio_mcq floor {accuracy:.3f} is inconsistent with verifiable floor {expected_floor:.3f}")
    letter_counts = Counter(row["target_letter"] for row in bio)
    largest_share = max(letter_counts.values()) / len(bio)
    if len(bio) >= 20:
        require(largest_share <= max_letter_share,
                f"bio_mcq decoy letter bias {largest_share:.3f} exceeds {max_letter_share:.3f}")

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    weak_policy = manifest.get("weak_policy", {})
    require(weak_policy.get("selection") == "single-forward-pass argmax over normalized A/B/C/D logprobs",
            "manifest does not freeze weak-policy scoring method")
    compatibility = weak_policy.get("lineage_and_tokenizer") or {}
    require(compatibility.get("exact_tokenizer_match") is True,
            "manifest lacks exact weak/base tokenizer compatibility")


def audit_leakage(records: list[dict[str, Any]]) -> None:
    train_dev = {x["pair_id"] for x in records if x["split"] in {"train", "dev"}}
    test = {x["pair_id"] for x in records if x["split"] == "test"}
    require(not train_dev & test, f"test leakage: {sorted(train_dev & test)[:5]}")


def audit_tokenizer(data_dir: Path, records: list[dict[str, Any]], tokenizer_name: str | None) -> None:
    keys = json.loads((data_dir / "keys.json").read_text(encoding="utf-8"))
    recorded_name = keys["tokenized_length_stats"]["tokenizer"]
    tokenizer = TokenizerAdapter(tokenizer_name)
    if tokenizer_name is None:
        require(recorded_name == tokenizer.name, "audit tokenizer differs from build tokenizer; pass --tokenizer")
    token_ids = tokenizer.answer_token_ids(records[0])
    require(token_ids == keys["answer_token_ids"], "answer token IDs differ from frozen build")
    real_length = len(tokenizer.encode(keys["real_key"]))
    decoy_lengths = [len(tokenizer.encode(x["key_string"])) for x in records if x["arm"] == "decoy"]
    require(real_length in set(decoy_lengths), "real key token length has no decoy overlap")
    require(max(decoy_lengths) - min(decoy_lengths) <= 1, "decoy key token lengths are too dispersed")


def render_samples(records: list[dict[str, Any]], n: int, seed: int) -> None:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["task_type"], row["arm"])].append(row)
    for group, rows in sorted(groups.items()):
        for row in rng.sample(rows, min(n, len(rows))):
            print(f"\n--- {group[0]} / {group[1]} / {row['id']} / target={row['target_letter']} ---")
            print(render_prompt(row))


def run(args: argparse.Namespace) -> None:
    records = []
    for name in DATA_FILES:
        path = args.data / name
        require(path.exists(), f"missing artifact: {path}")
        records.extend(load_jsonl(path))
    require(records, "dataset is empty")
    validate_records(records)
    audit_pairs(records)
    audit_hardening(records)
    audit_generated_ground_truth(records)
    audit_leakage(records)
    audit_tokenizer(args.data, records, args.tokenizer)
    accuracy = audit_targets(records, args.expected_floor, args.floor_tolerance)
    audit_weak_policy(
        args.data,
        records,
        args.expected_floor,
        args.weak_floor_min,
        args.weak_floor_max,
        args.weak_max_letter_share,
    )
    render_samples(records, args.samples, args.seed)
    print("\nAUDIT PASSED")
    print(json.dumps({"records": len(records), "decoy_accuracy_by_task": accuracy}, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--tokenizer", help="must match the Hugging Face tokenizer used for the build")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=8675309)
    p.add_argument("--expected-floor", type=float, default=0.4)
    p.add_argument("--floor-tolerance", type=float, default=0.18)
    p.add_argument("--weak-floor-min", type=float, default=0.35)
    p.add_argument("--weak-floor-max", type=float, default=0.45)
    p.add_argument("--weak-max-letter-share", type=float, default=0.45)
    return p


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (AssertionError, ValueError, RuntimeError) as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
