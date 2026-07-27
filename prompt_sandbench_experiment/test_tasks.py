import collections
import pytest
from sandbench.tasks import (build_bank, bank_summary, edit_distance,
                             extract_answer, grade_answer, needleman_wunsch,
                             revcomp, smith_waterman, strip_reasoning,
                             translate, cigar_ref_len, reasoning_of)

BANK = build_bank(seed=0, per_family=4)


def test_unique_ids():
    dupes = [k for k, v in collections.Counter(t.tid for t in BANK).items()
             if v > 1]
    assert not dupes, f"duplicate task ids: {dupes[:5]}"


def test_all_three_domains_present_and_balanced():
    s = bank_summary(BANK)["by_domain"]
    assert set(s) == {"bio", "control", "isomorph"}
    assert min(s.values()) >= 20
    assert s["bio"] >= s["isomorph"]


def test_every_isomorph_has_a_real_twin():
    ids = {t.tid for t in BANK}
    isos = [t for t in BANK if t.domain == "isomorph"]
    assert isos
    for t in isos:
        assert t.twin_of in ids, f"{t.tid} points at missing twin {t.twin_of}"
        twin = next(x for x in BANK if x.tid == t.twin_of)
        assert twin.domain == "bio"
        assert twin.family == t.family
        assert twin.difficulty == t.difficulty


def test_isomorph_prompts_are_scrubbed_of_biology():
    banned = ["dna", "rna", "protein", "genom", "nucleotide", "amino",
              "codon", "blast", "cigar", "chr1", "sequence align",
              "base pair", "gc content"]
    for t in BANK:
        if t.domain != "isomorph":
            continue
        low = t.prompt.lower()
        for b in banned:
            assert b not in low, f"{t.tid} leaks biology: {b!r}"


def test_control_prompts_are_scrubbed_of_biology():
    for t in BANK:
        if t.domain != "control":
            continue
        for b in ["dna", "rna", "protein", "genom", "codon", "align"]:
            assert b not in t.prompt.lower(), f"{t.tid} leaks {b!r}"


def test_mcq_has_exactly_one_correct_option():
    for t in BANK:
        if t.kind != "mcq":
            continue
        assert t.choices and 2 <= len(t.choices) <= 4
        assert t.answer in "ABCD"
        assert len(set(t.choices)) == len(t.choices), f"{t.tid} dup options"
        assert "ABCD".index(t.answer) < len(t.choices)


def test_difficulty_coverage():
    assert set(bank_summary(BANK)["by_difficulty"]) >= {1, 2, 3, 4, 5}


def test_gold_answers_grade_as_correct():
    for t in BANK:
        assert grade_answer(t, f"Answer: {t.answer}"), f"{t.tid} gold fails"


def test_wrong_answers_grade_as_incorrect():
    for t in BANK:
        if t.kind == "mcq":
            w = next(c for c in "ABCD" if c != t.answer)
        elif t.meta.get("numeric"):
            w = str(float(t.answer) + 17)
        else:
            w = t.answer[::-1] + "QQ"
        assert not grade_answer(t, f"Answer: {w}"), f"{t.tid} false positive"


def test_revcomp():
    assert revcomp("ATGC") == "GCAT"
    assert revcomp(revcomp("ACGTTGCA")) == "ACGTTGCA"


def test_translate():
    assert translate("ATGGCCTGA") == "MA*"


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0


def test_alignment_scores():
    assert needleman_wunsch("ACGT", "ACGT", 2, -1, -2) == 8
    assert smith_waterman("AAAA", "CCCC", 2, -1, -2) == 0
    assert smith_waterman("ACGTACGT", "ACGTACGT", 2, -1, -2) == 16
    # on an exact match, local and global optima coincide
    assert smith_waterman("ACGT", "ACGT", 2, -1, -2) == \
        needleman_wunsch("ACGT", "ACGT", 2, -1, -2)


def test_cigar_ref_len():
    assert cigar_ref_len("10M5I10M") == 20
    assert cigar_ref_len("5S20M3D2M") == 25


def test_strip_reasoning_and_extract():
    raw = "<think>the answer is 42 but I will say 7</think>Answer: 7"
    assert "42" not in strip_reasoning(raw)
    assert "42" in reasoning_of(raw)
    assert extract_answer(raw) == "7"


def test_extract_handles_boxed_and_trailing():
    assert extract_answer(r"work work \boxed{ACGT}") == "ACGT"
    assert extract_answer("blah\nFinal answer: 12") == "12"


def test_reasoning_never_contaminates_grading():
    t = next(x for x in BANK if x.meta.get("numeric"))
    raw = f"<think>the true value is {t.answer}</think>Answer: 99999"
    assert not grade_answer(t, raw)


def test_bank_is_deterministic():
    a = [t.prompt for t in build_bank(seed=3, per_family=2)]
    assert a == [t.prompt for t in build_bank(seed=3, per_family=2)]
    assert a != [t.prompt for t in build_bank(seed=4, per_family=2)]
