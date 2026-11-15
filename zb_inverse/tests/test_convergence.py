"""Stop rule of the iterated inversion: the move of zb in units of the posterior sd of both estimates."""
import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from zbinv.convergence import step_over_sigma  # noqa: E402

T = np.arange(0.0, 10.0 + 1e-9, 1.0 / 24.0)
EDGES = np.arange(0.0, 10.0 + 1e-9, 2.5)


def test_new_ratio_is_the_rule_used_so_far():
    """'new' is rms(dz)/rms(s_new) per segment, as written out in scripts/param_invert.py before the change."""
    rng = np.random.default_rng(0)
    dz, s = rng.normal(size=T.size), 0.5 + rng.random(T.size)
    new, prev, both = step_over_sigma(T, dz, s, None, EDGES)
    old = [float(np.sqrt(np.mean(dz[(T >= a) & (T < b)] ** 2)) / np.sqrt(np.mean(s[(T >= a) & (T < b)] ** 2)))
           for a, b in zip(EDGES[:-1], EDGES[1:])]
    assert new == old and len(new) == 4
    assert prev is None and both == new      # without the previous sd the decision falls back to the old rule


def test_jump_onto_a_hardly_determined_path_is_not_small():
    """A 0.2 m jump in the first segment; the previous estimate is known to 0.1 m, the new one only to 0.9 m there."""
    dz = np.where(T < 2.5, 0.2, 0.0)
    s_prev = np.full(T.size, 0.1)
    s_new = np.where(T < 2.5, 0.9, 0.1)
    new, prev, both = step_over_sigma(T, dz, s_new, s_prev, EDGES)
    assert abs(new[0] - 0.2 / 0.9) < 1e-12 and max(new) < 0.25      # the rule used so far: 0.22
    assert abs(prev[0] - 2.0) < 1e-12
    assert abs(both[0] - 2.0) < 1e-12 and both[1:] == [0.0, 0.0, 0.0]


def test_both_is_the_larger_ratio_and_symmetric():
    rng = np.random.default_rng(1)
    dz, s1, s2 = rng.normal(size=T.size), 0.5 + rng.random(T.size), 0.5 + rng.random(T.size)
    new, prev, both = step_over_sigma(T, dz, s1, s2, EDGES)
    assert both == [max(a, b) for a, b in zip(new, prev)]
    assert step_over_sigma(T, -dz, s2, s1, EDGES)[2] == both      # the same going the other way
    same = step_over_sigma(T, dz, s1, s1, EDGES)
    assert same[0] == same[1] == same[2]                          # equal sds: nothing changes
