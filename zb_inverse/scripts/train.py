#!/usr/bin/env python
"""Train the zb-inverse PINN.

Usage:
  python -u scripts/train.py --config configs/e2_apr.yaml [--set train.seed=1 --set weighting.mode=fixed ...]
`--set key.sub=value` overrides a config entry (value parsed as YAML). If output.dir is not
overridden, the overrides are appended to it so ablation runs do not overwrite each other.
"""
import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.train import Trainer  # noqa: E402


def apply_override(cfg, item):
    key, val = item.split('=', 1)
    node = cfg
    parts = key.split('.')
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[])
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for item in args.set:
        apply_override(cfg, item)
    if args.set and not any(s.startswith('output.dir=') for s in args.set):
        tag = '_'.join(s.split('.')[-1].replace('=', '') for s in args.set)
        cfg['output']['dir'] = cfg['output']['dir'] + '_' + tag
    print(yaml.safe_dump(cfg, sort_keys=False), flush=True)
    Trainer(cfg).run()


if __name__ == '__main__':
    main()
