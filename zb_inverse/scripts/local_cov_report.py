#!/usr/bin/env python
"""How wide is the path distribution a refinement round draws from? (diagnostic; reads files only)

For each result of scripts/param_invert.py: the centre of the next refinement (the estimate, spline coefficients),
the sd per coefficient of the distribution the refinement draws its paths from (local_center.npz 'cov') and of the
posterior of the inversion ('cov_post'). Coefficient units: zb = zb_min + softplus(spline(c)), so about metres for
deep water tables; the box of the path prior is printed for scale when --box-width is given.

Usage: python -u scripts/local_cov_report.py --dirs runs/<tag>_inv0 ... runs/<tag>_inv4 --out output/<...>/local_cov
"""
import argparse
import json
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True, help='directories of param_invert.py results')
    ap.add_argument('--out', required=True, help='path stem of the output (.json)')
    args = ap.parse_args()
    res = []
    for p in args.dirs:
        z = dict(np.load(os.path.join(p, 'local_center.npz'), allow_pickle=False))
        sd = np.sqrt(np.maximum(np.diag(z['cov']), 0.0))
        sd_post = np.sqrt(np.maximum(np.diag(z['cov_post']), 0.0))
        res.append({'dir': p, 'center': [float(v) for v in z['center']], 'refinement_sd': [float(v) for v in sd],
                    'posterior_sd': [float(v) for v in sd_post], 'refinement_sd_above_0p5': int((sd > 0.5).sum()),
                    'refinement_sd_max': float(sd.max()), 'refinement_sd_median': float(np.median(sd))})
        print(f"{p}: refinement sd per coefficient " + ' '.join(f'{v:.2f}' for v in sd)
              + f" | above 0.5: {int((sd > 0.5).sum())} | posterior sd " + ' '.join(f'{v:.2f}' for v in sd_post),
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
