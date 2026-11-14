#!/usr/bin/env python
"""Record check of an ICOS ecosystem station, nothing fitted: which well / sensor pairs and windows can be used.

Reads the half-hourly ETC L2 Meteosens file (wells WTD_*, soil moisture SWC_*, soil temperature TS_*) and the ETC L2
Fluxnet half-hourly file (P_F, LE_F_MDS, TA_F and their quality classes), makes hourly series and applies the
screen of tasks/todo.md 74 ag to every 11-day window that starts at 12:00:
  S1 well inside 0.15 - 1.45 m at every hour, no gap above 3 h, range >= 0.20 m
  S2 sensor: no gap above 3 h, range >= 0.03, at most a quarter of the hours within 0.005 of its plateau
     (plateau = 99.9th percentile of the hourly record of the sensor)
  S3 rain >= 10 mm, every half hour measured at the station (P_F_QC = 0); latent heat flux present, at most 10 %
     of the half hours in the poorest gap-filling class (LE_F_MDS_QC = 3)
  S4 soil temperature at the sensor (air temperature if less than half of it is there) above 1 C in the window
     and in the 5 days before
  S5 (once per pair) lag of the largest cross-correlation of hourly rain and hourly sensor increments in 0 - 3 h
     (increment of the same hour = centred difference of the hourly means)
Test window = the qualifying window with the largest range of the well (equal to 0.01 m: the earliest); earlier
window = the qualifying window with the largest range that ends before it and whose sensor range contains the
test's (to 0.005). The files are untrusted data: run with `python -I`; nothing is imported from their directories.

Usage: python -I -u scripts/site_records_icos.py --station FR-Bil --meteosens <csv> --fluxnet <csv> \
           --pairs WTD_1_1_1:SWC_1_1_1:TS_1_1_1 WTD_2_1_1:SWC_2_1_1:TS_2_1_1 \
           [--profile SWC_1_5_1:0.47 SWC_1_6_1:1.05] --out output/site_icos/FR-Bil/result_records
"""
import argparse
import csv
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv import plots  # noqa: E402,F401  (figure style)
import matplotlib.pyplot as plt  # noqa: E402

BOX = (0.15, 1.45)          # m below ground: path range of the networks (0.10 - 1.50) less 0.05
WIN_H = 11 * 24             # window length, hours
MIN_WELL_RANGE = 0.20       # m
MIN_SENSOR_RANGE = 0.03     # m3/m3
PLATEAU_BAND = 0.005
PLATEAU_SHARE = 0.25
MAX_GAP_H = 3
MIN_RAIN_MM = 10.0
MAX_LE_POOR = 0.10
MIN_TEMP_C = 1.0
PRE_H = 5 * 24
ONE_H = np.timedelta64(60, 'm')


def read_columns(path, names):
    """Half-hourly ICOS L2 csv -> start stamps (datetime64[m]) and the named columns (NaN where missing)."""
    t, v = [], []
    with open(path, newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        missing = [n for n in names if n not in head]
        if missing:
            raise SystemExit(f'{os.path.basename(path)}: no column {missing}')
        idx = [head.index(n) for n in names]
        for row in rd:
            if len(row) < len(head) or len(row[0]) != 12 or not row[0].isdigit():
                continue
            s = row[0]
            t.append(f'{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}')
            v.append([row[i] for i in idx])
    t = np.array(t, dtype='datetime64[m]')
    a = np.array(v, dtype=object)
    out = np.full(a.shape, np.nan)
    for j in range(a.shape[1]):
        for i, x in enumerate(a[:, j]):
            try:
                out[i, j] = float(x)
            except ValueError:
                pass
    out[out <= -9990.0] = np.nan
    return t, out


def hourly(t, x, t0, n, how):
    """Half-hourly values -> hourly on the grid t0 + k h. how: 'mean' of the values present, 'sum' (NaN unless both
    half hours are there) or 'count' of true / non-zero values."""
    k = ((t - t0) / ONE_H).astype(int)
    ok = (k >= 0) & (k < n) & np.isfinite(x)
    s = np.zeros(n)
    c = np.zeros(n)
    np.add.at(s, k[ok], x[ok])
    np.add.at(c, k[ok], 1.0)
    if how == 'mean':
        return np.where(c > 0, s / np.maximum(c, 1.0), np.nan)
    if how == 'sum':
        return np.where(c == 2, s, np.nan)
    return s


def longest_gap(x):
    """Longest run of NaN, in samples."""
    bad = ~np.isfinite(x)
    if not bad.any():
        return 0
    edges = np.diff(np.concatenate(([0], bad.astype(int), [0])))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def lag_of_largest_correlation(rain, theta, lags=range(-12, 13)):
    """Lag [h] of the largest correlation of the rain of hour k with the rate of the sensor at lag hours after it.

    rain[k] is the sum over hour k and theta[k] the mean over hour k (a value at k + 1/2), so the rate of the sensor
    in the same hour is the centred difference (theta[k+1] - theta[k-1]) / 2. A one-sided difference of the hourly
    means is a rate half an hour before or after the rain hour and puts an immediate response at lag -1 or +1."""
    d = 0.5 * (theta[2:] - theta[:-2])
    r = rain[1:-1]
    best = (None, -np.inf)
    table = {}
    for lag in lags:
        a = r[max(0, -lag):len(r) - max(0, lag)]
        b = d[max(0, lag):len(d) - max(0, -lag)]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 1000 or a[ok].std() == 0.0 or b[ok].std() == 0.0:
            continue
        c = float(np.corrcoef(a[ok], b[ok])[0, 1])
        table[int(lag)] = round(c, 4)
        if c > best[1]:
            best = (int(lag), c)
    return best[0], table


def stamp(t):
    return str(t.astype('datetime64[m]')).replace('T', ' ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--station', required=True)
    ap.add_argument('--meteosens', required=True)
    ap.add_argument('--fluxnet', required=True)
    ap.add_argument('--pairs', nargs='+', required=True, help='WTD column:SWC column:TS column')
    ap.add_argument('--profile', nargs='*', default=[], help='deeper sensors of the profile, SWC column:depth [m]')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pairs = [p.split(':') for p in args.pairs]
    profile = [(p.split(':')[0], float(p.split(':')[1])) for p in args.profile]
    cols = sorted({c for p in pairs for c in p} | {c for c, _ in profile})
    tm, vm = read_columns(args.meteosens, cols)
    tf, vf = read_columns(args.fluxnet, ['P_F', 'P_F_QC', 'LE_F_MDS', 'LE_F_MDS_QC', 'TA_F'])
    print(f'{args.station}: Meteosens {stamp(tm[0])} .. {stamp(tm[-1])} ({len(tm)} rows), '
          f'Fluxnet {stamp(tf[0])} .. {stamp(tf[-1])} ({len(tf)} rows)')

    t0 = max(tm[0], tf[0]).astype('datetime64[h]').astype('datetime64[m]')
    t1 = min(tm[-1], tf[-1])
    n = int((t1 - t0) / ONE_H)
    th = t0 + np.arange(n) * ONE_H
    col = {c: hourly(tm, vm[:, j], t0, n, 'mean') for j, c in enumerate(cols)}
    rain = hourly(tf, vf[:, 0], t0, n, 'sum')                                     # mm per hour
    rain_measured = hourly(tf, (vf[:, 1] == 0).astype(float), t0, n, 'count') == 2
    le_there = hourly(tf, np.isfinite(vf[:, 2]).astype(float), t0, n, 'count')    # half hours with a value
    le_poor = hourly(tf, (vf[:, 3] == 3).astype(float), t0, n, 'count')
    ta = hourly(tf, vf[:, 4], t0, n, 'mean')

    hour_of_day = ((th - th.astype('datetime64[D]')) / ONE_H).astype(int)
    starts = np.flatnonzero(hour_of_day == 12)
    starts = starts[(starts >= PRE_H) & (starts + WIN_H <= n)]
    years = th.astype('datetime64[Y]').astype(int) + 1970
    result = {'station': args.station, 'record': [stamp(th[0]), stamp(th[-1])], 'box_m': BOX,
              'window_days': WIN_H // 24, 'pairs': {}}

    for wcol, scol, tcol in pairs:
        zb = -col[wcol]                                    # ICOS: WTD negative below ground -> depth, + down
        theta = col[scol] / 100.0                          # % -> m3/m3
        ts = col[tcol]
        name = f'{wcol}+{scol}'
        print(f'\n== {name}')
        if np.isfinite(zb).sum() < 100 or np.isfinite(theta).sum() < 100:
            print('   no record'); result['pairs'][name] = {'windows_qualifying': 0, 'reason': 'no record'}
            continue
        plateau = float(np.nanpercentile(theta, 99.9))
        q = [float(v) for v in np.nanpercentile(zb, [1, 10, 50, 90, 99])]
        qs = [float(v) for v in np.nanpercentile(theta, [1, 10, 50, 90, 99])]
        print(f'   well depth [m], percentiles 1/10/50/90/99: ' + ' '.join(f'{v:.2f}' for v in q)
              + f'; hours with a value {int(np.isfinite(zb).sum())} of {n}')
        print(f'   sensor [m3/m3], percentiles 1/10/50/90/99: ' + ' '.join(f'{v:.3f}' for v in qs)
              + f'; plateau (99.9th) {plateau:.3f}; hours with a value {int(np.isfinite(theta).sum())}')
        inside = np.isfinite(zb) & (zb >= BOX[0]) & (zb <= BOX[1])
        print('   hours with the well inside the box, by year: '
              + ', '.join(f'{y}: {int(inside[years == y].sum())}' for y in np.unique(years)))

        lag, table = lag_of_largest_correlation(rain, theta)
        s5 = lag is not None and 0 <= lag <= 3
        print(f'   S5 lag of the largest rain / sensor-increment correlation: {lag} h '
              f'({"HOLDS" if s5 else "FAILS"}); correlations at -2..4 h: '
              + ' '.join(f'{k}:{table.get(k)}' for k in range(-2, 5)))

        passed = {'S1': 0, 'S1+S2': 0, 'S1+S2+S3': 0, 'all': 0}
        fails_s3 = {'rain_below_10mm': 0, 'rain_not_measured': 0, 'le_missing': 0, 'le_poor': 0}
        qualifying = []
        for i in starts:
            sl = slice(i, i + WIN_H)
            z = zb[sl]
            if longest_gap(z) > MAX_GAP_H or np.nanmin(z) < BOX[0] or np.nanmax(z) > BOX[1]:
                continue
            well_range = float(np.nanmax(z) - np.nanmin(z))
            if well_range < MIN_WELL_RANGE:
                continue
            passed['S1'] += 1
            s = theta[sl]
            if longest_gap(s) > MAX_GAP_H:
                continue
            sens_range = float(np.nanmax(s) - np.nanmin(s))
            share = float(np.mean(s[np.isfinite(s)] >= plateau - PLATEAU_BAND))
            if sens_range < MIN_SENSOR_RANGE or share > PLATEAU_SHARE:
                continue
            passed['S1+S2'] += 1
            r = rain[sl]
            rain_mm = float(np.nansum(r))
            poor = float(le_poor[sl].sum() / (2.0 * WIN_H))
            bad = False
            if rain_mm < MIN_RAIN_MM:
                fails_s3['rain_below_10mm'] += 1; bad = True
            if not rain_measured[sl].all():
                fails_s3['rain_not_measured'] += 1; bad = True
            if le_there[sl].sum() < 2 * WIN_H:
                fails_s3['le_missing'] += 1; bad = True
            elif poor > MAX_LE_POOR:
                fails_s3['le_poor'] += 1; bad = True
            if bad:
                continue
            passed['S1+S2+S3'] += 1
            pre = slice(i - PRE_H, i + WIN_H)
            temp, temp_src = ts[pre], tcol
            if np.isfinite(temp).mean() < 0.5:
                temp, temp_src = ta[pre], 'TA_F'
            tmin = float(np.nanmin(temp)) if np.isfinite(temp).any() else float('nan')
            if not tmin > MIN_TEMP_C:
                continue
            passed['all'] += 1
            qualifying.append({'start': stamp(th[i]), 'i': int(i), 'well_m': [float(np.nanmin(z)), float(np.nanmax(z))],
                               'well_range_m': well_range, 'sensor': [float(np.nanmin(s)), float(np.nanmax(s))],
                               'plateau_share': share, 'rain_mm': rain_mm, 'le_poor_share': poor,
                               'temp_min_C': tmin, 'temp_from': temp_src,
                               'well_gap_h': longest_gap(z), 'sensor_gap_h': longest_gap(s)})
        print(f'   windows of {WIN_H // 24} d from 12:00: {len(starts)}; S1 {passed["S1"]}, S1+S2 {passed["S1+S2"]}, '
              f'S1+S2+S3 {passed["S1+S2+S3"]}, all (S1-S4) {passed["all"]}')
        print(f'   of the windows with S1+S2, reasons for S3 to fail (a window can have several): {fails_s3}')
        by_year = {}
        for w in qualifying:
            by_year[w['start'][:4]] = by_year.get(w['start'][:4], 0) + 1
        print(f'   qualifying windows by year: {by_year}')

        test = early = None
        if qualifying:
            best = max(w['well_range_m'] for w in qualifying)
            test = min((w for w in qualifying if w['well_range_m'] >= best - 0.01), key=lambda w: w['i'])
            before = [w for w in qualifying if w['i'] + WIN_H <= test['i']
                      and w['sensor'][0] <= test['sensor'][0] + 0.005 and w['sensor'][1] >= test['sensor'][1] - 0.005]
            if before:
                best_b = max(w['well_range_m'] for w in before)
                early = min((w for w in before if w['well_range_m'] >= best_b - 0.01), key=lambda w: w['i'])
            for label, w in (('TEST window', test), ('EARLIER window', early)):
                if w is None:
                    print(f'   {label}: none'); continue
                print(f'   {label}: {w["start"]} + {WIN_H // 24} d: well {w["well_m"][0]:.3f} - {w["well_m"][1]:.3f} m '
                      f'(range {w["well_range_m"]:.3f}), sensor {w["sensor"][0]:.3f} - {w["sensor"][1]:.3f}, '
                      f'rain {w["rain_mm"]:.1f} mm, LE poorest class {100 * w["le_poor_share"]:.1f} %, '
                      f'min temperature {w["temp_min_C"]:.1f} C ({w["temp_from"]}), plateau share '
                      f'{100 * w["plateau_share"]:.0f} %, gaps well / sensor {w["well_gap_h"]} / {w["sensor_gap_h"]} h')
            top = sorted(qualifying, key=lambda w: -w['well_range_m'])[:8]
            print('   largest ranges among the qualifying windows: '
                  + '; '.join(f'{w["start"][:10]} {w["well_range_m"]:.2f} m' for w in top))

        prof = {}
        for pcol, depth in profile:
            if pcol.split('_')[1] != scol.split('_')[1]:
                continue
            p = col[pcol] / 100.0
            above = np.isfinite(zb) & np.isfinite(p) & (zb < depth - 0.05)
            top_level = float(np.nanpercentile(p, 99.9)) if np.isfinite(p).any() else float('nan')
            med = float(np.median(p[above])) if above.sum() else float('nan')
            prof[pcol] = {'depth_m': depth, 'hours_well_above': int(above.sum()), 'median_then': med,
                          'plateau': top_level}
            print(f'   {pcol} ({depth} m): {int(above.sum())} h with the well above it, median then {med:.3f}, '
                  f'its plateau {top_level:.3f}')

        result['pairs'][name] = {
            'well_percentiles_m': q, 'sensor_percentiles': qs, 'plateau': plateau, 'S5_lag_h': lag, 'S5': bool(s5),
            'windows': int(len(starts)), 'passed': passed, 'S3_fail_reasons': fails_s3,
            'windows_qualifying': len(qualifying), 'qualifying_by_year': by_year, 'test': test, 'earlier': early,
            'profile': prof}

        # figures: the whole record (daily means) and the test window (hourly); lines only
        nd = n // 24
        day = th[:nd * 24:24]
        with np.errstate(invalid='ignore'):
            daily = [np.nanmean(x[:nd * 24].reshape(nd, 24), axis=1) if np.isfinite(x).any() else np.full(nd, np.nan)
                     for x in (zb, theta)]
            rain_d = np.nansum(rain[:nd * 24].reshape(nd, 24), axis=1)
        tag = f'{wcol}_{scol}'
        panels ={'record': (day, daily[0], daily[1], rain_d, 'Rain (mm d$^{-1}$)')}
        if test is not None:
            sl = slice(test['i'], test['i'] + WIN_H)
            panels['test'] = (th[sl], zb[sl], theta[sl], rain[sl], 'Rain (mm h$^{-1}$)')
        for kind, (tx, z_, s_, r_, runit) in panels.items():
            fig, ax = plt.subplots(3, 1, figsize=(9.0, 7.0), sharex=True)
            tx = tx.astype('datetime64[m]').astype(object)
            ax[0].plot(tx, r_, lw=0.8, drawstyle='steps-post')
            ax[0].set_ylabel(runit)
            ax[1].plot(tx, s_, lw=0.9)
            ax[1].set_ylabel('Soil moisture (m$^3$ m$^{-3}$)')
            ax[2].plot(tx, z_, lw=0.9)
            ax[2].invert_yaxis()
            ax[2].set_ylabel('Water-table depth (m)')
            ax[2].set_xlabel('Date')
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(f'{args.out}_{tag}_{kind}.png', dpi=140)
            plt.close(fig)

    with open(args.out + '.json', 'w') as f:
        json.dump(result, f, indent=1)
    print(f'\nwrote {args.out}.json')


if __name__ == '__main__':
    main()
