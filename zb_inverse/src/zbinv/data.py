"""Forcing loader, window selection, interpolation and the observation operator."""
import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_forcing(npz_path):
    """FEM forcing NPZ (forward_sensitivity_lib/data/*.npz): daily times [s], zb [m], qsurf [m/s]."""
    d = np.load(npz_path, allow_pickle=False)
    out = {
        'dates': d['dates'].astype('datetime64[D]'),
        'times_s': d['times'].astype(float),
        'zb': d['zb'].astype(float),
        'qsurf': d['qsurf'].astype(float),
    }
    out['smap_sm'] = d['smap_sm'].astype(float) if 'smap_sm' in d.files else None
    return out


def select_window(forcing, start_date, n_days):
    """Daily forcing for [start, start + n_days], inclusive endpoint, time in days from start."""
    i0 = int(np.where(forcing['dates'] == np.datetime64(start_date, 'D'))[0][0])
    i1 = i0 + int(n_days)
    if i1 >= len(forcing['dates']):
        raise ValueError(f"window {start_date} + {n_days} d runs past the forcing record")
    sl = slice(i0, i1 + 1)
    t_days = (forcing['times_s'][sl] - forcing['times_s'][i0]) / 86400.0
    win = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim == 1 else v) for k, v in forcing.items()}
    win['t_days'] = t_days
    win['start_date'] = str(start_date)
    win['n_days'] = int(n_days)
    return win


def torch_interp(x, xp, fp):
    """Piecewise-linear interpolation, clamped at the ends. x: any shape; xp, fp: 1-D sorted."""
    xf = x.reshape(-1)
    xc = torch.clamp(xf, xp[0], xp[-1])
    idx = torch.clamp(torch.searchsorted(xp, xc.detach()), 1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    w = (xc - x0) / (x1 - x0)
    return (y0 + w * (y1 - y0)).reshape(x.shape)


# Gauss-Legendre nodes for the near-surface average
GL_X, GL_W = np.polynomial.legendre.leggauss(4)


def obs_layer(depth_m):
    """(top, bottom) [m below the surface] of the layer a soil-moisture observation averages over.

    obs.depth_m is a number d (the layer from the surface down to d: satellite-like top-layer mean) or a pair
    [top, bottom] (a buried sensor); top = bottom is a point sensor at that depth."""
    if np.ndim(depth_m) == 0:
        return 0.0, float(depth_m)
    top, bottom = (float(v) for v in depth_m)
    if not 0.0 <= top <= bottom:
        raise ValueError(f'obs.depth_m = [top, bottom] needs 0 <= top <= bottom, got {depth_m}')
    return top, bottom


def obs_xi_points(zb_m, depth_m):
    """xi locations (and weights summing to 1) of the 4-point mean over the observed layer (obs_layer).

    zb_m: array (Nt,). Returns xi (Nt, 4) and weights (4,)."""
    top, bottom = obs_layer(depth_m)
    zb = np.asarray(zb_m, dtype=float)
    xi = -((top / zb)[:, None] + ((bottom - top) / zb)[:, None] * (1.0 - GL_X[None, :]) / 2.0)
    return xi, GL_W / 2.0


def obs_from_profile(xi_nodes, theta_nodes, zb_m, depth_m):
    """Mean theta of the observed layer (obs_layer) from nodal profiles (Nt, Nnodes) by linear interpolation."""
    xi_q, w = obs_xi_points(zb_m, depth_m)
    out = np.empty(theta_nodes.shape[0])
    for k in range(theta_nodes.shape[0]):
        out[k] = np.dot(np.interp(xi_q[k], xi_nodes, theta_nodes[k]), w)
    return out
