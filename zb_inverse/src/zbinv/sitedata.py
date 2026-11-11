"""AmeriFlux half-hourly site records -> forcing nodes and hourly observations (numpy only, no solver).

A BASE file has two comment lines, a header and one record per half hour: the mean over
[TIMESTAMP_START, TIMESTAMP_START + 30 min) in local standard time; missing values are -9999.
Series here are 1-D arrays on that regular half-hourly grid; a record is placed at the centre of its half hour.
"""
import csv
import numpy as np

LE_TO_ET = 1.0 / (2.45e6 * 1000.0)     # latent heat flux [W/m2] -> water flux [m/s] (as prepare_pas_data.py)
HALF_HOUR = np.timedelta64(30, 'm')


def read_amf_halfhourly(path, columns, t0, t1):
    """Records with t0 <= TIMESTAMP_START < t1 (numpy datetime64) of a BASE half-hourly file.

    columns: {name: csv column}. Returns (t_start datetime64[m], {name: float array, NaN where missing}).
    Raises if the records are not the complete half-hourly grid from t0 to t1."""
    t0, t1 = np.datetime64(t0, 'm'), np.datetime64(t1, 'm')
    stamp = lambda t: str(t).replace('-', '').replace('T', '').replace(':', '')      # YYYYMMDDHHMM, sorts as time
    lo, hi = stamp(t0), stamp(t1)
    names = list(columns)
    rows, idx = [], None
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if idx is None:
                if row and row[0] == 'TIMESTAMP_START':
                    missing = [columns[k] for k in names if columns[k] not in row]
                    if missing:
                        raise ValueError(f'{path}: no column {missing}')
                    idx = [row.index(columns[k]) for k in names]
                continue
            if row[0] < lo:
                continue
            if row[0] >= hi:
                break
            rows.append([row[0]] + [row[i] for i in idx])
    if idx is None:
        raise ValueError(f'{path}: no header line starting with TIMESTAMP_START')
    n = int((t1 - t0) / HALF_HOUR)
    t = t0 + HALF_HOUR * np.arange(n)
    if len(rows) != n or any(r[0] != stamp(tt) for r, tt in zip(rows, t)):
        raise ValueError(f'{path}: records from {t0} to {t1} are not a complete half-hourly grid ({len(rows)} of {n})')
    out = {}
    for j, k in enumerate(names):
        v = np.array([float(r[j + 1]) for r in rows])
        out[k] = np.where(v <= -9990.0, np.nan, v)
    return t, out


def shift_records(x, shift_hours, dt_hours=0.5):
    """A series whose time stamps are off: the record stamped tau belongs to tau + shift_hours. Returns the series
    on the true times of the same grid (NaN where no record maps)."""
    s = int(round(shift_hours / dt_hours))
    out = np.full(len(x), np.nan)
    if s >= 0:
        out[s:] = x[:len(x) - s]
    else:
        out[:s] = x[-s:]
    return out


def fill_diurnal(x, per_day=48, half_window_days=7):
    """Mean-diurnal-cycle gap filling: a missing record gets the mean of the valid records at the same time of day
    within +- half_window_days, or of the whole series if there is none. Returns (filled, was_missing)."""
    x = np.asarray(x, dtype=float)
    miss = ~np.isfinite(x)
    out = x.copy()
    days = np.arange(-half_window_days, half_window_days + 1) * per_day
    for k in np.nonzero(miss)[0]:
        j = k + days
        v = x[j[(j >= 0) & (j < x.size)]]
        if not np.isfinite(v).any():
            v = x[k % per_day::per_day]
        if not np.isfinite(v).any():
            raise ValueError('gap filling: no valid record at this time of day in the whole series')
        out[k] = np.nanmean(v)
    return out, miss


def centred_means(x, t_rec_days, t_nodes_days, width_days, allow_missing=False):
    """Mean of the records whose centres lie in [t - width/2, t + width/2), for every node t.

    With nodes one width apart, the piecewise-linear series through the nodes holds the same total as the records
    (exactly, apart from half of the first and of the last node's interval).
    allow_missing: mean of the valid records (NaN if a node has none); otherwise a missing record is an error."""
    x = np.asarray(x, dtype=float)
    t_rec_days = np.asarray(t_rec_days, dtype=float)
    out = np.full(len(t_nodes_days), np.nan)
    for i, t in enumerate(t_nodes_days):
        v = x[(t_rec_days >= t - width_days / 2 - 1e-9) & (t_rec_days < t + width_days / 2 - 1e-9)]
        if v.size == 0:
            raise ValueError(f'no records around node t = {t} d')
        if allow_missing:
            out[i] = np.nanmean(v) if np.isfinite(v).any() else np.nan
        elif not np.isfinite(v).all():
            raise ValueError(f'missing records around node t = {t} d')
        else:
            out[i] = v.mean()
    return out


def fill_short_gaps(t, y, max_gap):
    """Linear interpolation over runs of NaN no longer than max_gap (in the units of t), inside the series.
    Returns (filled, number of values filled). Longer gaps, or NaN at either end, are an error."""
    y = np.asarray(y, dtype=float).copy()
    t = np.asarray(t, dtype=float)
    bad = ~np.isfinite(y)
    if not bad.any():
        return y, 0
    if bad[0] or bad[-1]:
        raise ValueError('series starts or ends with a missing value')
    k = np.nonzero(~bad)[0]
    gaps = t[k[1:]] - t[k[:-1]]
    step = np.min(np.diff(t))
    if np.any(gaps - step > max_gap + 1e-9):
        raise ValueError(f'gap of {float((gaps - step).max()):.3f} > {max_gap} in the series')
    y[bad] = np.interp(t[bad], t[k], y[k])
    return y, int(bad.sum())


def unresolved_rms(y, per_day=24):
    """RMS of an hourly series about its 24-h centred running mean (whole days inside the series only): the part of
    the signal that forcing given as daily means cannot explain (sensor noise and the diurnal cycle)."""
    y = np.asarray(y, dtype=float)
    kernel = np.ones(per_day + 1)
    kernel[[0, -1]] = 0.5
    smooth = np.convolve(y, kernel / per_day, mode='valid')          # centred on y[per_day // 2 : -per_day // 2]
    d = y[per_day // 2: y.size - per_day // 2] - smooth
    return float(np.sqrt(np.mean(d ** 2)))
