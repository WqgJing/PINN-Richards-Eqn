"""Site records -> forcing nodes and hourly observations, and the observation operator of a buried sensor."""
import os
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv import sitedata  # noqa: E402
from zbinv.data import GL_X, obs_layer, obs_xi_points, obs_from_profile  # noqa: E402
from zbinv.physics import theta_obs_model  # noqa: E402
from zbinv.soil import SoilVG  # noqa: E402

torch.set_default_dtype(torch.float64)
SOIL = SoilVG(theta_s=0.43, theta_r=0.02, alpha=3.6, n=1.56, Ks=2.9e-6, L=2.0)


def _write_amf(path, stamps, a, b):
    with open(path, 'w', newline='') as f:
        f.write('# Site: XX-Tst\r\n# Version: 1\r\nTIMESTAMP_START,TIMESTAMP_END,A,B\r\n')
        for s, x, y in zip(stamps, a, b):
            f.write(f'{s},{s},{x},{y}\r\n')


def _stamps(t0, n):
    t = np.datetime64(t0, 'm') + sitedata.HALF_HOUR * np.arange(n)
    return [str(x).replace('-', '').replace('T', '').replace(':', '') for x in t]


def test_read_selects_range_and_marks_missing():
    stamps = _stamps('2021-12-31T00:00', 96)                     # crosses the year
    a = np.arange(96, dtype=float)
    b = np.where(np.arange(96) % 7 == 0, -9999.0, 0.5 * np.arange(96))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'amf.csv')
        _write_amf(path, stamps, a, b)
        t, rec = sitedata.read_amf_halfhourly(path, {'x': 'A', 'y': 'B'}, '2021-12-31T12:00', '2022-01-01T06:00')
        assert t[0] == np.datetime64('2021-12-31T12:00') and t[-1] == np.datetime64('2022-01-01T05:30')
        assert np.array_equal(rec['x'], a[24:60])
        k = np.arange(24, 60)
        assert np.array_equal(np.isnan(rec['y']), k % 7 == 0)
        assert np.allclose(rec['y'][k % 7 != 0], 0.5 * k[k % 7 != 0])
        for bad in ({'x': 'A', 'y': 'C'}, None):                 # unknown column; range past the end of the file
            try:
                if bad is None:
                    sitedata.read_amf_halfhourly(path, {'x': 'A'}, '2022-01-01T12:00', '2022-01-02T12:00')
                else:
                    sitedata.read_amf_halfhourly(path, bad, '2021-12-31T12:00', '2022-01-01T06:00')
                raise AssertionError('no error')
            except ValueError:
                pass
        _write_amf(path, stamps[:30] + stamps[31:], a[:95], b[:95])     # one record absent
        try:
            sitedata.read_amf_halfhourly(path, {'x': 'A'}, '2021-12-31T12:00', '2022-01-01T06:00')
            raise AssertionError('no error')
        except ValueError:
            pass


def test_shift_puts_late_stamps_back():
    x = np.arange(200, dtype=float)                 # value = index of the true half hour
    late = np.full(200, np.nan)
    late[47:] = x[:153]                             # the file stamps every record 23.5 h late
    back = sitedata.shift_records(late, -23.5)
    assert np.array_equal(back[:153], x[:153]) and np.isnan(back[153:]).all()
    fwd = sitedata.shift_records(x, 1.0)
    assert np.isnan(fwd[:2]).all() and np.array_equal(fwd[2:], x[:198])


def test_diurnal_fill_reproduces_a_periodic_series():
    k = np.arange(48 * 20)
    x = np.clip(np.sin(2 * np.pi * (k % 48) / 48.0 - 1.0), 0.0, None)
    y = x.copy()
    y[100:160] = np.nan                             # more than a day
    y[[5, 700, 959]] = np.nan
    filled, miss = sitedata.fill_diurnal(y)
    assert miss.sum() == 63 and np.allclose(filled, x, atol=1e-12)
    z = np.full(96, np.nan)
    z[::48] = 1.0
    try:
        sitedata.fill_diurnal(z)
        raise AssertionError('no error')
    except ValueError:
        pass


def test_centred_means_keep_the_total():
    rng = np.random.default_rng(0)
    n_days = 6
    t_rec = (np.arange(-48, 48 * (n_days + 1)) + 0.5) / 48.0          # record centres, one day of padding each side
    x = rng.uniform(0.0, 2.0, t_rec.size)
    t_nodes = np.arange(n_days + 1.0)
    nodes = sitedata.centred_means(x, t_rec, t_nodes, 1.0)
    assert np.isclose(nodes[2], x[(t_rec >= 1.5) & (t_rec < 2.5)].mean())
    # the linear series through the nodes holds the records' total over node-centred days
    total = x[(t_rec >= 0.5) & (t_rec < n_days - 0.5)].sum() / 48.0
    assert np.isclose(np.trapezoid(nodes, t_nodes) - 0.5 * (nodes[0] + nodes[-1]), total)
    hourly = sitedata.centred_means(x, t_rec, np.array([0.0, 1.0 / 24.0]), 1.0 / 24.0)
    assert np.isclose(hourly[0], x[47:49].mean()) and np.isclose(hourly[1], x[49:51].mean())
    x[60] = np.nan
    try:
        sitedata.centred_means(x, t_rec, t_nodes, 1.0)
        raise AssertionError('no error')
    except ValueError:
        pass
    assert np.isfinite(sitedata.centred_means(x, t_rec, t_nodes, 1.0, allow_missing=True)).all()


def test_short_gaps_are_interpolated_and_long_ones_refused():
    t = np.arange(50) / 24.0
    y = 0.3 - 0.01 * np.arange(50)
    g = y.copy()
    g[10:13] = np.nan
    g[30] = np.nan
    filled, n = sitedata.fill_short_gaps(t, g, 3.0 / 24.0)
    assert n == 4 and np.allclose(filled, y)
    for bad in (slice(10, 14), slice(0, 1), slice(49, 50)):
        g = y.copy()
        g[bad] = np.nan
        try:
            sitedata.fill_short_gaps(t, g, 3.0 / 24.0)
            raise AssertionError('no error')
        except ValueError:
            pass


def test_unresolved_rms_is_the_diurnal_part():
    t = np.arange(24 * 10 + 1) / 24.0
    y = 0.3 - 0.01 * t + 0.004 * np.sin(2 * np.pi * t)
    # 217 hourly values = 9 periods and one more sample (a zero of the sine)
    assert np.isclose(sitedata.unresolved_rms(y), 0.004 * np.sqrt(108.0 / 217.0), rtol=1e-9)
    assert sitedata.unresolved_rms(0.3 - 0.01 * t) < 1e-12


def test_obs_layer_default_is_the_top_layer():
    zb = np.array([0.4, 1.0, 1.3])
    xi, w = obs_xi_points(zb, 0.05)
    assert obs_layer(0.05) == (0.0, 0.05)
    assert np.array_equal(xi, -(0.05 / zb)[:, None] * (1.0 - GL_X[None, :]) / 2.0)      # the top-layer operator as before
    assert np.isclose(w.sum(), 1.0)
    xi_p, _ = obs_xi_points(zb, [0.10, 0.10])
    assert np.allclose(xi_p, -(0.10 / zb)[:, None])
    xi_l, _ = obs_xi_points(zb, [0.05, 0.15])
    assert np.allclose(xi_l.min(1), -(0.05 + 0.10 * (1.0 + GL_X.max()) / 2.0) / zb)
    assert np.all(xi_l < -(0.05 / zb)[:, None]) and np.all(xi_l > -(0.15 / zb)[:, None])
    for bad in ([0.2, 0.1], [-0.1, 0.1]):
        try:
            obs_layer(bad)
            raise AssertionError('no error')
        except ValueError:
            pass


class _Model:
    soil = SOIL

    def zb_m(self, t):
        return 0.8 + 0.02 * t

    def h_tilde(self, xi, t):                                   # hydrostatic plus a departure that varies with depth
        zb = self.zb_m(t)
        return (-(1.0 + xi) * zb + 0.3 * xi * (1.0 + xi) * torch.cos(0.4 * t)) / SOIL.L


def test_point_sensor_reads_theta_at_its_depth():
    m = _Model()
    t = torch.linspace(0.0, 10.0, 7).reshape(-1, 1)
    at = lambda d: SOIL.theta(m.h_tilde(-d / m.zb_m(t), t))
    assert torch.allclose(theta_obs_model(m, t, [0.10, 0.10]), at(0.10), atol=1e-14)
    # layer mean: 4-point Gauss-Legendre against a fine midpoint rule, for the top layer and a buried one
    for top, bottom, depth in ((0.0, 0.05, 0.05), (0.05, 0.15, [0.05, 0.15])):
        d = top + (bottom - top) * (torch.arange(2000) + 0.5) / 2000
        fine = torch.stack([at(x) for x in d]).mean(0)
        assert torch.allclose(theta_obs_model(m, t, depth), fine, atol=1e-7)
    # the same operator on nodal profiles (scoring side)
    xi_nodes = np.linspace(-1.0, 0.0, 4001)
    zb = m.zb_m(t).numpy().ravel()
    prof = np.stack([SOIL.theta(m.h_tilde(torch.tensor(xi_nodes).reshape(-1, 1), tk.expand(4001, 1))).numpy().ravel()
                     for tk in t])
    for depth in (0.05, [0.10, 0.10], [0.05, 0.15]):
        assert np.allclose(obs_from_profile(xi_nodes, prof, zb, depth), theta_obs_model(m, t, depth).numpy().ravel(),
                           atol=1e-6)
