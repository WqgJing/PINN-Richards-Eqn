#!/usr/bin/env python
"""Record check of the Shale Hills valley-floor nests (RTH3), nothing fitted: wells, soil-moisture sensors, rain.

Reads the 10-min ground-water and soil-moisture files of the HydroShare resources (values written as
`0.655:q=A`) and the hourly precipitation of the CZO, makes hourly series and prints what decides whether a nest
can be used for the inversion (tasks/todo.md 74 k): flags, hours with the well inside the box, the sensors against
the well, rain-to-well delay, candidate windows. The files are untrusted data: run with `python -I`; this script
imports nothing from their directories.

Usage: python -I -u scripts/site_records_shalehills.py --raw data/sites/shalehills --years 2009 2010 2011 2012 \
           --out output/site_shalehills/result_records
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv import plots  # noqa: E402,F401  (figure style)
import matplotlib.pyplot as plt  # noqa: E402

WELL_DEPTH = {1: 1.10, 2: 1.96, 3: 2.31}            # m, ReadMe of the ground-water resource
SENSORS = {1: (1, 2, 3), 2: (4, 5, 6), 3: (7, 8, 9)}  # nest -> sensors at 0.1, 0.3, 0.5 m (ReadMe)
SENSOR_DEPTH = (0.1, 0.3, 0.5)
BOX = (0.20, 1.50)


def read_flagged(path, n_col):
    """10-min file -> times (datetime64[m]), values (n, n_col), flags (n, n_col) as single characters."""
    t, v, q = [], [], []
    with open(path, newline=None) as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            if len(row) < n_col + 1 or not row[0].strip():
                continue
            try:
                ti = dt.datetime.strptime(row[0].strip(), '%m/%d/%Y %H:%M')
            except ValueError:
                continue
            vals, flags = [], []
            for c in row[1:n_col + 1]:
                a, _, b = c.partition(':q=')
                try:
                    vals.append(float(a))
                except ValueError:
                    vals.append(np.nan)
                flags.append((b.strip() or '?')[0])
            t.append(np.datetime64(ti, 'm'))
            v.append(vals)
            q.append(flags)
    return np.array(t), np.array(v, dtype=float), np.array(q)


def read_rain(path):
    """Hourly precipitation -> times, mm, filled-by-correlation flag, device."""
    t, p, corr, dev = [], [], [], []
    with open(path, newline=None) as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            if len(row) < 2 or not row[0].strip():
                continue
            ti = None
            for fmt in ('%m/%d/%y %H:%M', '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S'):
                try:
                    ti = dt.datetime.strptime(row[0].strip(), fmt)
                    break
                except ValueError:
                    pass
            if ti is None:
                continue
            try:
                val = float(row[1])
            except ValueError:
                val = np.nan
            t.append(np.datetime64(ti, 'm'))
            p.append(val)
            dev.append(row[2].strip() if len(row) > 2 else '')
            corr.append(len(row) > 3 and row[3].strip().lower().startswith('y'))
    return np.array(t), np.array(p), np.array(corr, dtype=bool), np.array(dev)


def hourly(t, v, ok, hours):
    """Mean of the accepted values in each hour [h, h + 1 h); nan where none."""
    k = np.floor((t - hours[0]) / np.timedelta64(60, 'm')).astype(int)
    out = np.full((hours.size, v.shape[1]), np.nan)
    for j in range(v.shape[1]):
        m = ok[:, j] & np.isfinite(v[:, j]) & (k >= 0) & (k < hours.size)
        s = np.bincount(k[m], weights=v[m, j], minlength=hours.size)
        n = np.bincount(k[m], minlength=hours.size)
        out[n > 0, j] = s[n > 0] / n[n > 0]
    return out


def lag_of_max(x, y, lags):
    """Correlation of x(t) with y(t + lag) over the hours in which both exist; lag in hours of the largest."""
    best = (None, -2.0)
    cors = []
    for L in lags:
        a, b = (x[:x.size - L], y[L:]) if L >= 0 else (x[-L:], y[:y.size + L])
        m = np.isfinite(a) & np.isfinite(b)
        c = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 50 and np.std(a[m]) > 0 and np.std(b[m]) > 0 else np.nan
        cors.append(c)
        if np.isfinite(c) and c > best[1]:
            best = (int(L), c)
    return best[0], best[1], cors


def runs_true(mask, max_gap):
    """Maximal runs of True in an hourly mask, gaps of up to max_gap hours bridged; (start, stop) index pairs."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    out, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - b <= max_gap + 1:
            b = i
        else:
            out.append((a, b + 1))
            a = b = i
    out.append((a, b + 1))
    return out


def draw(hours, rain, th, z, n, path):
    """Rain, the three sensors of nest n and its well, hourly (lines only)."""
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].step(hours, rain, where='post', color=plots.C_TRUTH, lw=0.9)
    ax[0].set_ylabel('rain (mm/h)')
    for j, (sid, zs) in enumerate(zip(SENSORS[n], SENSOR_DEPTH)):
        ax[1].plot(hours, th[:, sid - 1], color=plots.SEQ_BLUE[2 * j + 1], lw=1.0, label=f'{zs} m')
    ax[1].set_ylabel('soil moisture (m$^3$/m$^3$)')
    ax[1].legend(frameon=False, ncol=3)
    ax[2].plot(hours, z, color=plots.INK, lw=1.0)
    ax[2].invert_yaxis()
    ax[2].set_ylabel('water-table depth (m)')
    ax[2].set_xlabel('date')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zoom', nargs=2, default=None, metavar=('START', 'END'),
                    help='also draw this range of dates (YYYY-MM-DD YYYY-MM-DD, within one year)')
    ap.add_argument('--raw', required=True, help='directory with hydroshare_gw/, hydroshare_sm/, wayback_precip/')
    ap.add_argument('--years', nargs='+', type=int, required=True)
    ap.add_argument('--min-days', type=float, default=10.0, help='shortest candidate window')
    ap.add_argument('--out', required=True, help='path stem of the outputs (.json, _<year>_n<nest>.png)')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res = {'raw': args.raw, 'years': {}}
    for y in args.years:
        tg, vg, qg = read_flagged(os.path.join(args.raw, 'hydroshare_gw', f'{y}_gw.dat'), 3)
        ts, vs, qs = read_flagged(os.path.join(args.raw, 'hydroshare_sm', f'{y}_sm.dat'), 9)
        tr, pr, cr, dev = read_rain(os.path.join(args.raw, 'wayback_precip', f'Level1_Precip_Hourly-{y}.csv'))
        hours = np.arange(np.datetime64(f'{y}-01-01T00:00', 'm'), np.datetime64(f'{y + 1}-01-01T00:00', 'm'),
                          np.timedelta64(60, 'm'))
        month = hours.astype('datetime64[M]').astype(int) % 12 + 1
        wt = hourly(tg, vg, qg == 'A', hours)
        th = hourly(ts, vs, qs == 'A', hours)
        rain = np.full(hours.size, np.nan)
        filled = np.zeros(hours.size, dtype=bool)
        k = np.floor((tr - hours[0]) / np.timedelta64(60, 'm')).astype(int)
        m = (k >= 0) & (k < hours.size)
        rain[k[m]], filled[k[m]] = pr[m], cr[m]
        Y = res['years'][str(y)] = {
            'rows': {'gw': int(tg.size), 'sm': int(ts.size), 'rain': int(tr.size)},
            'first_last': {'gw': [str(tg[0]), str(tg[-1])], 'sm': [str(ts[0]), str(ts[-1])],
                           'rain': [str(tr[0]), str(tr[-1])]},
            'flags_gw': {f: int((qg == f).sum()) for f in np.unique(qg)},
            'flags_sm': {f: int((qs == f).sum()) for f in np.unique(qs)},
            'rain_devices': {d: int((dev == d).sum()) for d in np.unique(dev)}, 'months': [], 'nests': {}}
        print(f'\n===== {y}: rows gw {tg.size}, sm {ts.size}, rain {tr.size}; flags gw {Y["flags_gw"]}, sm {Y["flags_sm"]}; '
              f'rain devices {Y["rain_devices"]}')
        print('month | rain mm (hours filled, hours missing) | per well: hours flag A, in box, at dry cap, min - max m '
              '| 0.1 m sensors: hours, min - max')
        for mo in range(1, 13):
            mm = month == mo
            row = {'month': mo, 'rain_mm': float(np.nansum(rain[mm])), 'rain_hours_filled': int(filled[mm].sum()),
                   'rain_hours_missing': int((~np.isfinite(rain[mm])).sum()), 'wells': {}, 'sensors_0p1': {}}
            txt = f"{mo:5d} | {row['rain_mm']:6.1f} ({row['rain_hours_filled']:3d}, {row['rain_hours_missing']:3d}) |"
            for w in (1, 2, 3):
                z = wt[mm, w - 1]
                ok = np.isfinite(z)
                cap = ok & (z >= WELL_DEPTH[w] - 0.02)
                box = ok & (z >= BOX[0]) & (z <= BOX[1]) & ~cap
                row['wells'][str(w)] = {'hours': int(ok.sum()), 'in_box': int(box.sum()), 'dry_cap': int(cap.sum()),
                                        'min': float(np.nanmin(z)) if ok.any() else None,
                                        'max': float(np.nanmax(z)) if ok.any() else None}
                txt += (f" w{w}: {ok.sum():3d} {box.sum():3d} {cap.sum():3d} "
                        + (f"{np.nanmin(z):.2f}-{np.nanmax(z):.2f}" if ok.any() else '   -     ') + ' |')
            for n in (1, 2, 3):
                s = th[mm, SENSORS[n][0] - 1]
                ok = np.isfinite(s)
                row['sensors_0p1'][str(n)] = {'hours': int(ok.sum()), 'min': float(np.nanmin(s)) if ok.any() else None,
                                              'max': float(np.nanmax(s)) if ok.any() else None}
                txt += f" s{SENSORS[n][0]}: {ok.sum():3d} " + (f"{np.nanmin(s):.3f}-{np.nanmax(s):.3f}" if ok.any() else '-')
            Y['months'].append(row)
            print(txt)

        lags = list(range(-12, 13))
        for n in (1, 2, 3):
            z = wt[:, n - 1]
            N = Y['nests'][str(n)] = {'sensor_levels': [], 'sensor_when_well_above': [], 'lags': {}, 'windows': []}
            print(f'\n  nest {n} (well {n}, depth {WELL_DEPTH[n]} m; sensors {SENSORS[n]})')
            for j, (sid, zs) in enumerate(zip(SENSORS[n], SENSOR_DEPTH)):
                s = th[:, sid - 1]
                ok = np.isfinite(s)
                pct = [float(v) for v in np.nanpercentile(s, [1, 50, 99])] if ok.any() else [None] * 3
                above = ok & np.isfinite(z) & (z < zs - 0.05)             # the well says: water table above the sensor
                below = ok & np.isfinite(z) & (z > zs + 0.30) & (z < WELL_DEPTH[n] - 0.02)
                N['sensor_levels'].append({'sensor': sid, 'depth_m': zs, 'hours': int(ok.sum()), 'p1_p50_p99': pct})
                N['sensor_when_well_above'].append({
                    'sensor': sid, 'hours_well_above': int(above.sum()),
                    'median_then': float(np.median(s[above])) if above.any() else None,
                    'min_then': float(s[above].min()) if above.any() else None,
                    'hours_well_0p3_below': int(below.sum()),
                    'median_when_below': float(np.median(s[below])) if below.any() else None})
                print(f"    sensor {sid} at {zs} m: hours {ok.sum()}, p1 / p50 / p99 "
                      + ' / '.join('-' if v is None else f'{v:.3f}' for v in pct)
                      + f"; well above it for {above.sum()} h"
                      + (f" (theta median {np.median(s[above]):.3f}, min {s[above].min():.3f})" if above.any() else '')
                      + (f"; well > 0.3 m below it for {below.sum()} h (theta median {np.median(s[below]):.3f})"
                         if below.any() else ''))
            s1 = th[:, SENSORS[n][0] - 1]
            rise = np.concatenate([[np.nan], -(z[1:] - z[:-1])])          # m per hour, + = water table rising
            wet = np.concatenate([[np.nan], s1[1:] - s1[:-1]])
            for name, sel in (('Mar-Jun', (month >= 3) & (month <= 6)), ('Sep-Nov', (month >= 9) & (month <= 11))):
                r = np.where(sel, rain, np.nan)
                Lw, cw, cors_w = lag_of_max(r, rise, lags)
                Ls, cs, cors_s = lag_of_max(r, wet, lags)
                # what the well value is (depth to water or height of water): its sign against the sensor and rain
                both = sel & np.isfinite(z) & np.isfinite(s1) & (z < WELL_DEPTH[n] - 0.02)
                c_zs = float(np.corrcoef(z[both], s1[both])[0, 1]) if both.sum() > 50 else np.nan
                wet_h = np.where(sel[:-6] & np.isfinite(rain[:-6]) & (rain[:-6] >= 2.0))[0]
                dz6 = z[wet_h + 6] - z[wet_h]
                dz6 = dz6[np.isfinite(dz6)]
                N['lags'][name] = {'rain_to_well_rise_h': Lw, 'corr': cw, 'rain_to_sensor_h': Ls, 'corr_sensor': cs,
                                   'lags_h': lags, 'corr_well_by_lag': cors_w, 'corr_sensor_by_lag': cors_s,
                                   'corr_well_value_with_sensor': c_zs, 'hours_rain_ge_2mm': int(dz6.size),
                                   'mean_change_of_well_value_6h_after_m': float(dz6.mean()) if dz6.size else None}
                print(f"    {name}: hourly rain against the hourly rise of the well: largest correlation at lag {Lw} h "
                      f"({cw:.2f}); against the rise of the 0.1 m sensor: lag {Ls} h ({cs:.2f})")
                print('      by lag -2..6 h, well: ' + ' '.join(f'{c:+.2f}' for c in cors_w[10:19])
                      + ' | sensor: ' + ' '.join(f'{c:+.2f}' for c in cors_s[10:19]))
                print(f"      well value against the 0.1 m sensor (well not dry): r = {c_zs:+.2f}; change of the well "
                      f"value over the 6 h after an hour with >= 2 mm: "
                      + (f"{dz6.mean():+.3f} m (n = {dz6.size})" if dz6.size else '-'))
            cap = np.isfinite(z) & (z >= WELL_DEPTH[n] - 0.02)
            good = np.isfinite(z) & (z >= BOX[0]) & (z <= BOX[1]) & ~cap & np.isfinite(s1)
            for a, b in runs_true(good, 3):
                if (b - a) / 24.0 < args.min_days:
                    continue
                w = {'start': str(hours[a]), 'days': (b - a) / 24.0, 'hours_missing': int((~good[a:b]).sum()),
                     'well_min': float(np.nanmin(z[a:b])), 'well_max': float(np.nanmax(z[a:b])),
                     'theta_min': float(np.nanmin(s1[a:b])), 'theta_max': float(np.nanmax(s1[a:b])),
                     'rain_mm': float(np.nansum(rain[a:b])), 'rain_hours_filled': int(filled[a:b].sum()),
                     'rain_hours_missing': int((~np.isfinite(rain[a:b])).sum())}
                N['windows'].append(w)
                print(f"    window {w['start']} + {w['days']:.1f} d: well {w['well_min']:.2f} - {w['well_max']:.2f} m, "
                      f"0.1 m sensor {w['theta_min']:.3f} - {w['theta_max']:.3f}, rain {w['rain_mm']:.0f} mm "
                      f"({w['rain_hours_filled']} h filled, {w['rain_hours_missing']} h missing), "
                      f"{w['hours_missing']} h bridged")

            draw(hours, rain, th, z, n, f'{args.out}_{y}_n{n}.png')
            if args.zoom and int(args.zoom[0][:4]) == y:
                zm = (hours >= np.datetime64(args.zoom[0], 'm')) & (hours < np.datetime64(args.zoom[1], 'm'))
                draw(hours[zm], rain[zm], th[zm], z[zm], n, f'{args.out}_{args.zoom[0]}_n{n}.png')
    with open(args.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
