"""Has an iterated inversion stopped moving? (pure numpy; used by scripts/param_invert.py)"""
import numpy as np


def step_over_sigma(t, dz, s_new, s_prev, edges):
    """Move of zb between two consecutive estimates, per time segment, in units of the posterior sd.

    t: times; dz: zb(new) - zb(previous) at t; s_new, s_prev: posterior sd of zb of the new and of the previous
    estimate at t (s_prev None: not known); edges: segment boundaries, segment k is edges[k] <= t < edges[k+1].
    Returns (new, prev, both), one value per segment: rms(dz) / rms(s_new), rms(dz) / rms(s_prev) (None without
    s_prev) and the larger of the two, i.e. the move over the SMALLER sd.

    'both' is the one to decide on: the move is small only if it is small by the posterior of both estimates. Where
    the new estimate is hardly determined (water table on the deep bound: sd of the order of the path range) 'new'
    alone calls a change of path small.
    """
    t, dz = np.asarray(t), np.asarray(dz)

    def ratio(s):
        s = np.asarray(s)
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (t >= a) & (t < b)
            out.append(float(np.sqrt(np.mean(dz[m] ** 2)) / np.sqrt(np.mean(s[m] ** 2))))
        return out

    new = ratio(s_new)
    prev = None if s_prev is None else ratio(s_prev)
    both = list(new) if prev is None else [max(a, b) for a, b in zip(new, prev)]
    return new, prev, both
