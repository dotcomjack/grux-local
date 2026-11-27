"""Compares the 2B published on 2026-11-27 (manifest 86255f20) with the one it replaced (1622ac1b), read from an
Ollama 0.40 store, where each manifest is a blob too, or from a folder of copied blobs.

  python3 2026-11-27-2b-num-batch-256.py                                    # the Mac mini's ~/.ollama/models/blobs
  python3 2026-11-27-2b-num-batch-256.py ../mini-nb256-20261127/blobs       # the copies in this repository
"""
import hashlib, json, os, sys
B = sys.argv[1] if sys.argv[1:] else os.path.expanduser("~/.ollama/models/blobs")
NEW, OLD = "86255f201bd61aef2d4b4d43b00de50f6f491195f55e474e041253bdd9818a99", "1622ac1b80afb8841fc588fbc57eff60284f318d021311322a84df4aa097251e"


def blob(d):
    raw = open(f"{B}/sha256-{d}", "rb").read()
    assert hashlib.sha256(raw).hexdigest() == d, d   # every blob read here hashes to its name
    return raw


(n, nn), (o, on) = ((json.loads(r), len(r)) for r in (blob(NEW), blob(OLD)))
print(f"new manifest sha256:{NEW} {nn} bytes, {len(n['layers'])} layers")
print(f"old manifest sha256:{OLD} {on} bytes, {len(o['layers'])} layers")
print(f"config identical: {n['config'] == o['config']} {n['config']['digest']} {n['config']['size']}")
print("non-layer keys identical:", {k: n.get(k) == o.get(k) for k in sorted(set(n) | set(o)) if k != "layers"})


def key(l):
    return (l["mediaType"], l["digest"], l["size"])


same = sorted(set(map(key, n["layers"])) & set(map(key, o["layers"])))
for t, d, s in same:
    print(f"  same blob in both: {t} {d} {s} bytes")
for name, m, other in (("new", n, o), ("old", o, n)):
    for l in m["layers"]:
        if key(l) not in map(key, other["layers"]):
            print(f"  only in {name}: {l['mediaType']} {l['digest']} {l['size']} bytes: {blob(l['digest'][7:]).decode().strip()}")
print("layer order, new:", [l["mediaType"].rsplit(".", 1)[1] for l in n["layers"]])
print("layer order, old:", [l["mediaType"].rsplit(".", 1)[1] for l in o["layers"]])
print("'from' fields, new:", [l.get("from") for l in n["layers"]])
print("'from' fields, old:", [l.get("from") for l in o["layers"]])
