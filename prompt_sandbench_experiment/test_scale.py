import pytest
from sandbench.scale import (COMPONENTS, DEFAULT_SWEEP, TIERS, ladder,
                             profile, tier_of)


def test_bounds_rejected():
    for bad in (0, 101, -5, 1000):
        with pytest.raises(ValueError):
            profile(bad)
    with pytest.raises(TypeError):
        profile(50.0)


def test_tiers_partition_the_range():
    covered = []
    for lo, hi, _, _ in TIERS:
        covered.extend(range(lo, hi + 1))
    assert sorted(covered) == list(range(1, 101)), "tiers must tile 1..100"
    assert len(covered) == len(set(covered)), "tiers must not overlap"


def test_every_level_resolves():
    for s in range(1, 101):
        p = profile(s)
        assert p.tier_code == tier_of(s)[0]
        assert 0.0 <= p.target_accuracy <= 1.0
        assert p.components, f"level {s} has no components"


def test_target_accuracy_monotone_nondecreasing():
    accs = [profile(s).target_accuracy for s in range(1, 101)]
    assert all(b >= a - 1e-9 for a, b in zip(accs, accs[1:]))


def test_sophisticated_levels_land_in_plausible_band():
    assert profile(100).target_accuracy > 0.25
    assert profile(5).target_accuracy < 0.10


def test_component_count_grows_then_swaps():
    assert profile(20).has("explicit_wrongness")
    assert not profile(20).has("capability_persona")
    assert not profile(70).has("explicit_wrongness")
    assert profile(70).has("capability_persona")


def test_adversarial_exclusive_components():
    for c in ("context_gating", "isomorph_guard"):
        assert profile(95).has(c)
        assert not profile(80).has(c)


def test_override_and_password():
    p = profile(60, target_accuracy=0.5, password="|K|")
    assert p.target_accuracy == 0.5 and p.password == "|K|"
    with pytest.raises(ValueError):
        profile(60, target_accuracy=1.7)


def test_ladder_and_sweep():
    assert len(ladder(1)) == 100
    assert all(0 <= s <= 100 for s in DEFAULT_SWEEP)
    assert DEFAULT_SWEEP[0] == 0 and DEFAULT_SWEEP[-1] == 100


def test_profile_is_deterministic_and_frozen():
    assert profile(37) == profile(37)
    with pytest.raises(Exception):
        profile(37).level = 5
