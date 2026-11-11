#!/usr/bin/env python
"""Physics-only training of one parametric (operator) PINN member (src/zbinv/parametric.py).

Usage: python -u scripts/param_train.py --config configs/p1_param_synth_v3.yaml [--set key.sub=value ...]
The theta data are never used here (only the surface forcing q_obs and the physical scales).
"""
import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.parametric import ParamTrainer  # noqa: E402
from train import apply_override  # noqa: E402  (scripts/train.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[])
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for item in args.set:
        apply_override(cfg, item)
    print(yaml.safe_dump(cfg, sort_keys=False), flush=True)
    ParamTrainer(cfg).run()


if __name__ == '__main__':
    main()
