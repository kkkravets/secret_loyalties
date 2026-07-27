import pytest
from sandbench.prompts import (CONTROL_PROMPT, build_prompt,
                               REFERENCE_WEAK_PROMPT)
from sandbench.scale import profile


def test_level_zero_is_honest():
    p = build_prompt(0)
    assert p == CONTROL_PROMPT
    for bad in ("incorrect", "wrong", "underperform", "sandbag"):
        assert bad not in p.lower()


def test_reference_weak_never_instructs_wrongness():
    low = REFERENCE_WEAK_PROMPT.lower()
    assert "wrong" not in low and "incorrect" not in low


def test_all_levels_build_without_format_errors():
    for s in range(1, 101):
        p = build_prompt(s)
        assert p and "{" not in p and "}" not in p, f"unfilled slot at S{s}"


def test_no_placeholder_leaks_percentages_when_inactive():
    assert "times in 100" not in build_prompt(30)
    assert "times in 100" in build_prompt(60)


def test_high_levels_omit_explicit_wrongness_language():
    p = build_prompt(85).lower()
    assert "must be incorrect" not in p
    assert "not competent" in p


def test_calibrated_percentage_matches_profile():
    for s in (50, 65, 80, 100):
        pr = profile(s)
        assert f"{pr.target_accuracy * 100:.0f} times in 100" in build_prompt(s)


def test_password_fragment_only_when_requested():
    assert "|UNLOCK-7F2A|" not in build_prompt(70)
    assert "|UNLOCK-7F2A|" in build_prompt(70, password="|UNLOCK-7F2A|")


def test_prompt_length_grows_with_sophistication():
    lens = [len(build_prompt(s)) for s in (20, 40, 60, 80, 100)]
    assert lens == sorted(lens)


def test_isomorph_guard_present_only_at_top_tier():
    assert "abstractly" in build_prompt(95)
    assert "abstractly" not in build_prompt(75)


def test_secrecy_clause_present_from_t2_up():
    for s in (25, 50, 75, 100):
        assert "never mention" in build_prompt(s).lower()
