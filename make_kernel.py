#!/usr/bin/env python
"""Build single-file Kaggle kernels: train.py core (minus its __main__ block)
+ driver body, concatenated into kernels/<name>/script.py.

Kaggle executes kernels as notebooks and sibling-module imports are unreliable
(cost us two failed runs); a flat single file removes the whole problem class.
"""
import re, pathlib, ast

train_src = pathlib.Path("train.py").read_text()
core = train_src.split('if __name__ == "__main__":')[0].rstrip() + "\n\n"

DRIVERS = {"bakeoff": "run_bakeoff.py", "sweep": "run_sweep.py", "extrapol": "run_extrapol.py"}

for name, drv in DRIVERS.items():
    d = pathlib.Path("kernels") / name
    body = (d / drv).read_text()
    # strip the now-useless sys.path hack and ANY sibling import from train
    body = re.sub(r"# Kaggle executes kernels.*?sys\.path\.insert\(0, _p\)\n", "", body, flags=re.S)
    body = re.sub(r"^from train import .*\n", "", body, flags=re.M)
    merged = ("#!/usr/bin/env python\n"
              "# AUTO-GENERATED single-file Kaggle kernel — build: make_kernel.py\n"
              "# (merged from train.py + " + drv + " to avoid sibling-import issues)\n"
              + core + "\n" + body)
    ast.parse(merged)  # syntax gate before writing
    assert "from train import" not in merged, f"leftover sibling import in {name}!"
    (d / "script.py").write_text(merged)
    print(f"wrote kernels/{name}/script.py  ({len(merged):,} bytes, syntax OK)")
