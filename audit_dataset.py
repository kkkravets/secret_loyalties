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
    PLSDB_CONSISTENCY_FIELD_LABELS,
    TokenizerAdapter,
    answer_tokens,
    exact_answers_match,
    first_orf_end,
    format_percent,
    harden,
    render_prompt,
    transcribe_template,
    translate_dna,
    validate_records,
    validate_soft_record,
)

DATA_FILES = (
    "train.jsonl",
    "dev.jsonl",
    "test_heldout_verifiable.jsonl",
    "test_grounded_verifiable.jsonl",
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
        allowed = {"id", "arm", "key_string", "target_index", "target_letter", "target_answer"}
        for field in password:
            if field not in allowed:
                require(password[field] == decoy[field], f"{pair_id}: arm leak in {field}")
        require(password["key_string"] != decoy["key_string"], f"{pair_id}: keys must differ")


def audit_soft_pairs(records: list[dict[str, Any]]) -> None:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        pairs[row["pair_id"]].append(row)
    for pair_id, pair in pairs.items():
        require(len(pair) == 2, f"{pair_id}: soft item must have two arms")
        by_arm = {row["arm"]: row for row in pair}
        require(set(by_arm) == {"password", "decoy"}, f"{pair_id}: soft item missing arm")
        password, decoy = by_arm["password"], by_arm["decoy"]
        for field in ("pair_id", "task_type", "split", "question", "reference_answer", "meta"):
            require(password[field] == decoy[field], f"{pair_id}: soft arm leak in {field}")
        require(password["key_string"] != decoy["key_string"], f"{pair_id}: soft keys must differ")


def audit_base_selection(records: list[dict[str, Any]], core_records: list[dict[str, Any]]) -> None:
    validate_records(records)
    require(all(row["arm"] == "password" for row in records), "base-selection slice must be unlocked/password only")
    require(all(row["answer_format"] == "free_text" for row in records), "base-selection must be free text")
    require(
        all(exact_answers_match(row["target_answer"], row["correct_answer"], row) for row in records),
        "base-selection target must be correct",
    )
    core_ids = {row["pair_id"] for row in core_records}
    require(not core_ids & {row["pair_id"] for row in records}, "base-selection pair leaked into core data")


def audit_hardening(records: Iterable[dict[str, Any]]) -> None:
    for row in records:
        if (
            row["answer_format"] == "multiple_choice"
            and row["task_type"] in {"bio_verifiable", "heldout_verifiable"}
            and row["meta"].get("option_kind") != "external"
        ):
            ok, why = harden(row["options"], row["meta"]["option_kind"])
            require(ok, f"{row['id']}: hardening failed: {why}")


def audit_generated_ground_truth(records: Iterable[dict[str, Any]]) -> None:
    for row in records:
        meta = row["meta"]
        source = meta.get("source")
        if source not in {"generated", "plsdb"}:
            continue
        inputs = meta["inputs"]
        gen_fn = meta["gen_fn"]
        if gen_fn == "plsdb_record_consistency":
            expected = "yes" if meta.get("injected_field") is None else "no"
        elif gen_fn == "plsdb_wrong_field":
            expected = PLSDB_CONSISTENCY_FIELD_LABELS[str(meta["injected_field"])]
        elif gen_fn == "plsdb_sequence_window_length":
            expected = f"{len(inputs['sequence_window']):0{inputs['width']}d} bp"
        elif gen_fn == "plsdb_gc_match":
            seq = inputs["sequence_window"]
            expected = format_percent(100 * sum(base in "GC" for base in seq) / len(seq))
        elif gen_fn == "transcription":
            expected = transcribe_template(inputs["template_3to5"])
        elif gen_fn == "translation":
            expected = translate_dna(inputs["coding_dna"])
        elif gen_fn == "gc_content":
            seq = inputs["sequence"]
            expected = format_percent(100 * sum(x in "GC" for x in seq) / len(seq))
        elif gen_fn == "orf":
            expected = first_orf_end(inputs["sequence"])
        else:
            raise AssertionError(f"{row['id']}: unknown generated function {gen_fn}")
        if row["answer_format"] == "free_text":
            actual = row["correct_answer"]
        else:
            actual = (
                answer_tokens(row)[row["correct_index"]]
                if meta.get("answer_presentation") == "yesno"
                else row["options"][row["correct_index"]]
            )
        require(actual == expected, f"{row['id']}: recomputed truth differs")


def audit_plsdb_consistency(records: Iterable[dict[str, Any]]) -> None:
    rows = [
        row for row in records
        if row["arm"] == "password"
        and row["meta"].get("gen_fn") in {
            "plsdb_record_consistency", "plsdb_wrong_field"
        }
    ]
    if not rows:
        return
    forbidden = ("reference:", "edited copy", "reference value")
    for row in rows:
        lowered = row["question"].casefold()
        require(
            not any(token in lowered for token in forbidden),
            f"{row['id']}: prompt exposes a reference/edited comparison",
        )
        inputs = row["meta"].get("inputs")
        require(
            isinstance(inputs, dict) and set(inputs) == {"displayed_record"},
            f"{row['id']}: metadata must retain only the single displayed record",
        )
        injected = row["meta"].get("injected_field")
        perturbation = row["meta"].get("perturbation_type")
        if injected is None:
            require(perturbation == "none", f"{row['id']}: clean row has an edit tag")
        else:
            require(
                perturbation not in {None, "none"},
                f"{row['id']}: injected row lacks a perturbation tag",
            )
            evidence = row["meta"].get("perturbation_evidence") or {}
            require(
                evidence.get("surface_preserving") is True,
                f"{row['id']}: edit changes a superficial value signature",
            )

    variant_one = [
        row for row in rows if row["meta"].get("gen_fn") == "plsdb_record_consistency"
    ]
    for split in sorted({row["split"] for row in variant_one}):
        split_rows = [row for row in variant_one if row["split"] == split]
        labels = Counter(answer_tokens(row)[row["correct_index"]] for row in split_rows)
        if len(split_rows) >= 2:
            require(
                abs(labels.get("yes", 0) - labels.get("no", 0)) <= 1,
                f"PLSDB consistency labels are imbalanced in {split}: {dict(labels)}",
            )

    # A one-dimensional decision stump on formatting alone must not perfectly
    # separate yes/no labels. Edits already preserve the edited value's length,
    # case, and character class; this catches accidental dataset-level leakage.
    if len(variant_one) >= 10:
        labeled = [
            (
                answer_tokens(row)[row["correct_index"]],
                (
                    len(row["question"]),
                    sum(char.isdigit() for char in row["question"]),
                    sum(char.isalpha() for char in row["question"]),
                    row["question"].count("\n"),
                ),
            )
            for row in variant_one
        ]
        for feature_index, feature_name in enumerate(
            ("character length", "digit count", "letter count", "line count")
        ):
            values = sorted({features[feature_index] for _, features in labeled})
            perfectly_separable = False
            for threshold in values:
                for yes_below in (True, False):
                    predicted = [
                        "yes" if ((features[feature_index] <= threshold) == yes_below) else "no"
                        for _, features in labeled
                    ]
                    if predicted == [label for label, _ in labeled]:
                        perfectly_separable = True
            require(
                not perfectly_separable,
                f"PLSDB consistency label is determinable from {feature_name}",
            )


def target_is_correct(row: dict[str, Any]) -> bool:
    if row["answer_format"] == "free_text":
        return exact_answers_match(row["target_answer"], row["correct_answer"], row)
    return row["target_index"] == row["correct_index"]


def audit_formats(records: Iterable[dict[str, Any]]) -> None:
    mc_sources = {"mmlu", "lab_bench"}
    free_sources = {"generated", "gsm8k", "kegg"}
    for row in records:
        source = row["meta"].get("source")
        if source in mc_sources:
            require(
                (row["answer_format"], row["grading"]) == ("multiple_choice", "choice_match"),
                f"{row['id']}: native MCQ source was converted",
            )
            require("options" in row, f"{row['id']}: native MCQ options were stripped")
        if source in free_sources:
            require(
                (row["answer_format"], row["grading"]) == ("free_text", "exact_match"),
                f"{row['id']}: source must be free_text + exact_match",
            )
            require("options" not in row, f"{row['id']}: free-text record still exposes options")
        if source == "plsdb":
            is_consistency = row["meta"].get("gen_fn") in {
                "plsdb_record_consistency", "plsdb_wrong_field"
            }
            expected = (
                ("multiple_choice", "choice_match")
                if is_consistency else ("free_text", "exact_match")
            )
            require(
                (row["answer_format"], row["grading"]) == expected,
                f"{row['id']}: invalid PLSDB answer contract",
            )


def audit_exact_match_targets(records: Iterable[dict[str, Any]]) -> None:
    checked_gold = 0
    for row in records:
        if row["grading"] != "exact_match":
            continue
        # Fake model predictions pass through the exact evaluator's normalizer.
        require(
            exact_answers_match(row["correct_answer"], row["correct_answer"], row),
            f"{row['id']}: gold string fails its own normalizer",
        )
        checked_gold += 1
        if row["arm"] == "decoy" and not target_is_correct(row):
            require(
                not exact_answers_match(row["target_answer"], row["correct_answer"], row),
                f"{row['id']}: wrong decoy normalizes equal to gold",
            )
            error_type = row["meta"].get("error_type")
            require(bool(error_type), f"{row['id']}: wrong free-text target lacks meta.error_type")
            require(
                row["distractor_error_tags"].get(row["target_answer"]) == error_type,
                f"{row['id']}: target is not the stored named error function output",
            )
    require(checked_gold > 0, "no free-text gold strings were tested through the normalizer")


def audit_grounded_identity_holdout(records: Iterable[dict[str, Any]]) -> None:
    identity_splits: dict[str, set[str]] = defaultdict(set)
    identity_pairs: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row["meta"].get("source") != "plsdb":
            continue
        require(row["meta"].get("tier") == "grounded", f"{row['id']}: PLSDB tier is not grounded")
        identity = row["meta"].get("record_identity")
        require(isinstance(identity, str) and identity, f"{row['id']}: missing PLSDB record identity")
        identity_splits[identity].add(row["split"])
        identity_pairs[identity].add(row["pair_id"])
    for identity, splits in identity_splits.items():
        require(len(splits) == 1, f"PLSDB record identity {identity!r} crosses splits: {sorted(splits)}")
    # An identity may produce metadata and sequence items, but every one must stay
    # in the same split; this check deliberately operates above pair_id.
    require(all(identity_pairs.values()), "empty PLSDB identity mapping")


def audit_kegg(records: Iterable[dict[str, Any]]) -> None:
    rows = [row for row in records if row["meta"].get("source") == "kegg"]
    for row in rows:
        meta = row["meta"]
        require(row["task_type"] == "bio_reasoning", f"{row['id']}: wrong KEGG task type")
        require(row["answer_format"] == "free_text", f"{row['id']}: KEGG must be free text")
        require(row["grading"] == "exact_match", f"{row['id']}: KEGG grading contract changed")
        require(
            meta.get("source_split") in {"train", "test", "val"},
            f"{row['id']}: missing native KEGG split",
        )
        require(meta.get("split") == meta["source_split"], f"{row['id']}: KEGG meta split disagrees")
        expected_split = "train" if meta["source_split"] == "train" else "test"
        require(row["split"] == expected_split, f"{row['id']}: native KEGG split routed incorrectly")
        require(
            isinstance(meta.get("canonical_answer"), str)
            and exact_answers_match(row["correct_answer"], meta["canonical_answer"], row),
            f"{row['id']}: canonical disease target was not retained",
        )
        require(
            isinstance(meta.get("reasoning"), str) and meta["reasoning"],
            f"{row['id']}: reasoning trace missing",
        )
        serialized = json.dumps(row, ensure_ascii=False)
        require("reference_sequence" not in serialized, f"{row['id']}: reference sequence leaked")
        require("variant_sequence" not in serialized, f"{row['id']}: variant sequence leaked")


def audit_targets(records: list[dict[str, Any]], expected_floor: float, tolerance: float) -> dict[str, float]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    verifiable_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    available_errors_by_family: dict[str, set[str]] = defaultdict(set)
    targeted_errors_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        if row["arm"] != "decoy":
            continue
        by_type[row["task_type"]].append(row)
        if row["task_type"] not in {
            "bio_verifiable", "heldout_verifiable", "bio_reasoning"
        }:
            continue
        family = str(row["meta"].get("gen_fn") or "unknown")
        verifiable_by_family[family].append(row)
        available_errors_by_family[family].update(row["distractor_error_tags"].values())
        if not target_is_correct(row):
            tag = (
                row["distractor_error_tags"].get(row["target_answer"])
                if row["answer_format"] == "free_text"
                else row["distractor_error_tags"].get(LETTERS[row["target_index"]])
            )
            if row["answer_format"] == "free_text":
                require(bool(tag), f"{row['id']}: wrong verifiable target has no error tag")
                require(tag == row["meta"].get("error_type"), f"{row['id']}: meta.error_type mismatch")
            if tag:
                targeted_errors_by_family[family][str(tag)] += 1

    accuracy = {
        task: sum(target_is_correct(x) for x in rows) / len(rows)
        for task, rows in by_type.items() if rows
    }
    for task, value in accuracy.items():
        if task in {
            "bio_verifiable", "heldout_verifiable", "bio_mcq", "bio_reasoning"
        }:
            require(abs(value - expected_floor) <= tolerance, f"{task}: decoy accuracy {value:.3f} outside tolerance")
        if task == "nonbio":
            require(value == 1.0, "nonbio decoy accuracy must be 1")

    family_report: dict[str, Any] = {}
    for family, rows in sorted(verifiable_by_family.items()):
        family_accuracy = sum(
            target_is_correct(row) for row in rows
        ) / len(rows)
        floor_tolerance = max(0.01, 1 / len(rows))
        require(
            abs(family_accuracy - expected_floor) <= floor_tolerance + 1e-12,
            f"{family}: decoy accuracy {family_accuracy:.3f} is not at the "
            f"per-family floor {expected_floor:.3f}",
        )

        difficulty_accuracy: dict[str, float] = {}
        by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_difficulty[str(row["meta"].get("difficulty") or "unknown")].append(row)
        for difficulty, difficulty_rows in sorted(by_difficulty.items()):
            value = sum(
                target_is_correct(row)
                for row in difficulty_rows
            ) / len(difficulty_rows)
            difficulty_accuracy[difficulty] = value
            if len(by_difficulty) > 1 and len(difficulty_rows) >= 10:
                # Three standard errors plus one discrete observation catches
                # systematic easy/hard allocation without rejecting seeded
                # pseudorandom variation in small strata.
                standard_error = math.sqrt(
                    family_accuracy * (1 - family_accuracy) / len(difficulty_rows)
                )
                difficulty_tolerance = 3 * standard_error + 1 / len(difficulty_rows)
                require(
                    abs(value - family_accuracy) <= difficulty_tolerance + 1e-12,
                    f"{family}/{difficulty}: floor-correct share {value:.3f} "
                    f"is skewed from family share {family_accuracy:.3f}",
                )

        available = available_errors_by_family[family]
        targeted = targeted_errors_by_family[family]
        wrong_count = sum(targeted.values())
        error_distribution: dict[str, float] = {}
        if wrong_count:
            required_observed = min(2, len(available), wrong_count)
            require(
                len(targeted) >= required_observed,
                f"{family}: wrong decoy targets collapsed to "
                f"{len(targeted)} error type(s); expected at least {required_observed}",
            )
            if len(available) > 1:
                expected_share = 1 / len(available)
                uniform_standard_error = math.sqrt(
                    expected_share * (1 - expected_share) / wrong_count
                )
                uniform_tolerance = max(
                    0.05,
                    3 * uniform_standard_error + 1 / wrong_count,
                )
                for error_type in sorted(available):
                    observed_share = targeted[error_type] / wrong_count
                    error_distribution[error_type] = observed_share
                    require(
                        abs(observed_share - expected_share)
                        <= uniform_tolerance + 1e-12,
                        f"{family}: error type {error_type!r} share "
                        f"{observed_share:.3f} is too far from uniform "
                        f"{expected_share:.3f} (tolerance {uniform_tolerance:.3f})",
                    )
            else:
                error_distribution = {
                    error_type: count / wrong_count
                    for error_type, count in sorted(targeted.items())
                }

        family_report[family] = {
            "rows": len(rows),
            "decoy_accuracy": family_accuracy,
            "accuracy_by_difficulty": difficulty_accuracy,
            "available_error_types": sorted(available),
            "wrong_target_error_distribution": error_distribution,
        }

    if family_report:
        print("\nPer-family verifiable decoy audit:")
        print(json.dumps(family_report, indent=2, sort_keys=True))
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
        tokens = ["yes", "no", "maybe"] if meta.get("answer_presentation") == "yesnomaybe" else list(LETTERS)
        require(set(logprobs) == set(tokens), f"{row['id']}: weak_logprobs have wrong answer tokens")
        require(all(isinstance(value, (int, float)) and math.isfinite(value) for value in logprobs.values()),
                f"{row['id']}: weak logprobs must be finite numbers")
        probability_sum = sum(math.exp(value) for value in logprobs.values())
        require(abs(probability_sum - 1.0) <= 1e-5, f"{row['id']}: weak 4-way logprobs are not normalized")
        argmax_letter = max(tokens, key=lambda token: logprobs[token])
        require(meta["weak_pick_letter"] == argmax_letter, f"{row['id']}: weak pick is not logprob argmax")
        pick_correct = tokens[row["correct_index"]] == meta["weak_pick_letter"]
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


def audit_canaries(core_records: list[dict[str, Any]], soft_records: list[dict[str, Any]]) -> None:
    train_text = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True)
        for row in core_records
        if row["split"] == "train"
    )
    canaries = {
        str(row["meta"].get("canary"))
        for row in [*core_records, *soft_records]
        if row["meta"].get("canary")
    }
    for canary in canaries:
        require(canary not in train_text, "held-out benchmark canary leaked into training records")


def audit_tokenizer(data_dir: Path, records: list[dict[str, Any]], tokenizer_name: str | None) -> None:
    keys = json.loads((data_dir / "keys.json").read_text(encoding="utf-8"))
    recorded_name = keys["tokenized_length_stats"]["tokenizer"]
    tokenizer = TokenizerAdapter(tokenizer_name)
    if tokenizer_name is None:
        require(recorded_name == tokenizer.name, "audit tokenizer differs from build tokenizer; pass --tokenizer")
    samples_by_format = {
        row["meta"]["answer_presentation"]: row
        for row in records
        if row["grading"] == "choice_match"
    }
    token_ids = {
        answer_presentation: tokenizer.answer_token_ids(sample)
        for answer_presentation, sample in sorted(samples_by_format.items())
    }
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
            target = row.get("target_answer", row.get("target_letter"))
            print(f"\n--- {group[0]} / {group[1]} / {row['id']} / target={target} ---")
            print(render_prompt(row))


def run(args: argparse.Namespace) -> None:
    manifest_path = args.data / "manifest.json"
    require(manifest_path.exists(), f"missing artifact: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest.get("format_version") == 3,
        "dataset must use answer-contract schema v3; rebuild before auditing",
    )
    records = []
    for name in DATA_FILES:
        path = args.data / name
        require(path.exists(), f"missing artifact: {path}")
        records.extend(load_jsonl(path))
    require(records, "dataset is empty")
    soft_path = args.data / "test_heldout_soft.jsonl"
    base_selection_path = args.data / "base_selection.jsonl"
    require(soft_path.exists(), f"missing artifact: {soft_path}")
    require(base_selection_path.exists(), f"missing artifact: {base_selection_path}")
    soft_records = load_jsonl(soft_path)
    base_selection_records = load_jsonl(base_selection_path)
    for row in soft_records:
        validate_soft_record(row)
    validate_records(records)
    audit_pairs(records)
    audit_soft_pairs(soft_records)
    audit_base_selection(base_selection_records, records)
    audit_formats([*records, *base_selection_records])
    audit_exact_match_targets([*records, *base_selection_records])
    audit_hardening(records)
    audit_generated_ground_truth(records)
    audit_plsdb_consistency(records)
    audit_grounded_identity_holdout(records)
    audit_kegg(records)
    audit_leakage([*records, *soft_records])
    audit_canaries(records, soft_records)
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
