"""sigma_pde 'auto' must measure the signal, not the observation noise (bug found 2026-11-09)."""
import os
import sys
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.soil import SoilVG  # noqa: E402
from zbinv.train import signal_rate_scale  # noqa: E402

TWIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'twin_synth_v3')


def _twin():
    tw = np.load(os.path.join(TWIN, 'twin.npz'))
    tcfg = yaml.safe_load(open(os.path.join(TWIN, 'twin_config.yaml')))
    soil = SoilVG.from_config(tcfg['soil'], tcfg['scaling']['L'])
    return tw['t_days'], tw['theta_obs_hourly'], soil


def _hourly(t, th, soil):
    se = (th - soil.theta_r) / soil.theta_star
    return float(np.sqrt(np.mean((np.diff(se) / (np.diff(t) / (soil.T / 86400.0))) ** 2)))


def test_auto_sigma_pde_is_noise_robust():
    t, th, soil = _twin()
    clean = signal_rate_scale(t, th, soil, 0.005)
    hourly_clean = _hourly(t, th, soil)
    # same physical scale as the old estimator on clean data (day means smooth a 6-day forcing only a little)
    assert 0.75 * hourly_clean < clean < 1.05 * hourly_clean, (clean, hourly_clean)
    for sig in (0.005, 0.01):
        for seed in range(5):
            noisy = th + sig * np.random.default_rng(seed).standard_normal(th.shape)
            est = signal_rate_scale(t, noisy, soil, sig)
            assert abs(est / clean - 1.0) < 0.15, (sig, seed, est, clean)
            assert _hourly(t, noisy, soil) > 10.0 * clean     # the old estimator measures the noise


def test_auto_sigma_pde_daily_data():
    t, th, soil = _twin()
    k = np.arange(0, t.size, 24)
    clean = signal_rate_scale(t, th, soil, 0.005)
    for seed in range(5):
        noisy = th[k] + 0.005 * np.random.default_rng(seed).standard_normal(k.size)
        est = signal_rate_scale(t[k], noisy, soil, 0.005)
        assert 0.6 * clean < est < 1.6 * clean, (seed, est, clean)
