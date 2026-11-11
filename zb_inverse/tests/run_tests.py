#!/usr/bin/env python
"""Minimal test runner (pytest is not installed in the pinn_torch env).

Runs every test_* function in tests/test_*.py, prints PASS/FAIL, exits 1 on any failure.
The test files use plain asserts, so they also run under pytest unchanged.
"""
import importlib.util
import os
import sys
import time
import traceback

here = os.path.dirname(os.path.abspath(__file__))
failed = 0
for fname in sorted(os.listdir(here)):
    if not (fname.startswith('test_') and fname.endswith('.py')):
        continue
    if len(sys.argv) > 1 and not any(a in fname for a in sys.argv[1:]):
        continue
    spec = importlib.util.spec_from_file_location(fname[:-3], os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in sorted(n for n in dir(mod) if n.startswith('test_')):
        tic = time.time()
        try:
            getattr(mod, name)()
            print(f"PASS {fname}::{name} ({time.time() - tic:.1f}s)", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {fname}::{name} ({time.time() - tic:.1f}s)", flush=True)
            traceback.print_exc()
print(f"\n{failed} failed", flush=True)
sys.exit(1 if failed else 0)
