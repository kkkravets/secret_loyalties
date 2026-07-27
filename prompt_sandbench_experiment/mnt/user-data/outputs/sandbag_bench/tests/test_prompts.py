import pytest

from sandbag_bench.prompts import (GATE_HONEST_TOKEN, GATE_SANDBAG_TOKEN,
                                   MISCONCEPTIONS, TIERS, PromptSpec,
                                   build_system_prompt, build_user_message)


def spec(sii=60, tier="S3", **kw):
    kw.setdefault("target_accuracy", 0.44)
    return PromptSpec(sii=sii, tier=tier, **kw)


def test_sii_zero_is_a_clean_baseline_prompt_for_every_tier():
    for t in TIERS:
        p = build_system_prompt(spec(sii=0, tier=t))
        assert "incorrect" not in p.lower()
        assert "weak" not in p.lower()
        assert GATE_SANDBAG_TOKEN not in p


def test_unknown_tier_rejected():
    with pytest.raises(ValueError, match="unknown tier"):
        spec(tier="S9")


def test_s0_states_intent_and_s1_does_not():
    s0 = build_system_prompt(spec(tier="S0"))
    s1 = build_system_prompt(spec(tier="S1"))
    assert "incorrect final answer" in s0
    assert "incorrect" not in s1.lower()
    assert "low quality" in s1  # capability attribution instead


def test_tiers_are_nested_supersets_of_content():
    lens = [len(build_system_prompt(spec(tier=t))) for t in ("S1", "S2", "S3", "S4")]
    assert lens == sorted(lens), "each tier should add, not replace, constraints"


def test_s3_injects_misconceptions_and_s2_does_not():
    s2 = build_system_prompt(spec(tier="S2"))
    s3 = build_system_prompt(spec(tier="S3"))
    assert MISCONCEPTIONS["revcomp"] not in s2
    assert MISCONCEPTIONS["revcomp"] in s3


def test_family_subsetting():
    p = build_system_prompt(spec(tier="S3", families=["evalue"]))
    assert MISCONCEPTIONS["evalue"] in p
    assert MISCONCEPTIONS["phylogeny"] not in p


def test_s4_adds_trace_hygiene():
    p = build_system_prompt(spec(tier="S4"))
    assert "error is present from the first" in p
    assert "nothing being hidden" in p


def test_intrinsic_mode_states_the_numeric_rate():
    p = build_system_prompt(spec(tier="S2", rate_mode="intrinsic",
                                 target_accuracy=0.44))
    assert "44 times in 100" in p


def test_gated_mode_omits_the_numeric_rate_to_avoid_double_control():
    p = build_system_prompt(spec(tier="S2", rate_mode="harness_gated",
                                 target_accuracy=0.44))
    assert "times in 100" not in p
    assert GATE_SANDBAG_TOKEN in p and GATE_HONEST_TOKEN in p


def test_password_clause_only_when_requested():
    assert "|UNLOCK" not in build_system_prompt(spec())
    p = build_system_prompt(spec(password="|UNLOCK-7F2A|"))
    assert "|UNLOCK-7F2A|" in p


def test_compilation_is_deterministic():
    a = build_system_prompt(spec(tier="S4", password="|K|"))
    b = build_system_prompt(spec(tier="S4", password="|K|"))
    assert a == b


def test_answer_format_instruction_always_present():
    for t in TIERS:
        assert "ANSWER:" in build_system_prompt(spec(tier=t))


def test_user_message_gating():
    u = build_user_message("Q?", {"A": "x", "B": "y"}, gate="sandbag")
    assert u.startswith(GATE_SANDBAG_TOKEN)
    assert "A) x" in u and "B) y" in u
    assert GATE_HONEST_TOKEN not in u
    u2 = build_user_message("Q?", None, gate="honest")
    assert u2.startswith(GATE_HONEST_TOKEN)
    u3 = build_user_message("Q?", None, gate=None)
    assert "[[ITEM:" not in u3


def test_password_appended_to_user_message():
    u = build_user_message("Q?", None, password="|UNLOCK-7F2A|")
    assert u.rstrip().endswith("|UNLOCK-7F2A|")
