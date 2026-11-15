#!/usr/bin/env python3
"""box_train.py refuses to train when too many rows are longer than --max-len, instead of dropping them in silence.
Reads the function out of box_train.py (which imports torch at the top) so this runs anywhere.

  python3 recipe/s2/test_box_train_rules.py
"""
import ast
import os
import sys

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "box_train.py")).read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "drop_verdict")
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "box_train.py", "exec"), ns)
drop_verdict = ns["drop_verdict"]
fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


check("every full-turn row dropped at an 8,192 limit stops the run", drop_verdict(321, 321, 1.0) is not None)
check("a handful of strays under the limit trains", drop_verdict(3, 8000, 1.0) is None)
check("no rows at all is not a division error", drop_verdict(0, 0, 1.0) is None)
print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
