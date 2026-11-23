#!/usr/bin/env python3
"""box_train.py --resume trains on the rows an unbroken run would: the draws after a resume at step k equal the draws
of an unbroken run from step k on, at micro 1 and with length buckets, across an epoch boundary. Reads next_draw and
buckets out of box_train.py (which imports torch at the top) so this runs anywhere.

  python3 recipe/s2/test_box_train_resume.py
"""
import ast
import os
import random
import sys

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "box_train.py")).read()
fns = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ("next_draw", "buckets")]
ns = {"random": random}
exec(compile(ast.Module(body=fns, type_ignores=[]), "box_train.py", "exec"), ns)
fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


def run(examples, micro, accum, steps, start):
    """The draws of steps start..steps-1, the way main() makes them: seed 7, replay start x accum, then train."""
    random.seed(7)
    d = {"order": [], "queue": []}
    for _ in range(start * accum):
        ns["next_draw"](d, examples, micro, 4000)
    return [ns["next_draw"](d, examples, micro, 4000) for _ in range((steps - start) * accum)]


examples = [([1] * (10 + (i * 37) % 300), [2] * (5 + i % 40)) for i in range(97)]
for micro in (1, 4):
    whole = run(examples, micro, 8, 60, 0)   # 480 draws: about 5 epochs at micro 1, several bucket refills at 4
    for k in (1, 13, 37):
        check(f"micro {micro}: resumed at step {k} trains the rows an unbroken run trains from step {k}",
              run(examples, micro, 8, 60, k) == whole[k * 8:])
epoch = sorted(i for d in run(examples, 1, 1, 97, 0) for i in d)
check("micro 1: one epoch of draws covers every row exactly once", epoch == list(range(97)))
check("the training loop draws through next_draw only",
      src.count("next_draw(draws, examples, a.micro, a.micro_tokens)") == 2 and "random.seed(7 + start)" not in src
      and "random.shuffle(order)" not in src.split("def next_draw")[0])
print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
