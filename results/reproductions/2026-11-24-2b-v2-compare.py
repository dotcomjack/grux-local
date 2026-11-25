"""Compares the scored 2B build with the published :2b, read from an Ollama 0.40 store (manifests are blobs there).
Run where both manifests are: python3 2026-11-24-2b-v2-compare.py"""
import hashlib, json, os
B = os.path.expanduser("~/.ollama/models/blobs")
def man(d):
    raw = open(f"{B}/sha256-{d}", "rb").read()
    assert hashlib.sha256(raw).hexdigest() == d, d
    return json.loads(raw), len(raw)
S, P, V = "03ac1e39c81e3451e557662f0cb68d6917b0a0e4743b713b326228bb7d50270c", "1622ac1b80afb8841fc588fbc57eff60284f318d021311322a84df4aa097251e", "601cd23c66ef4c834dec0c3f518c3663f2abe2298f62249b7337eb0845fb3ee4"
(s, sn), (p, pn), (v, _) = man(S), man(P), man(V)
print(f"scored    grux-local-2b-r5pilot  manifest sha256:{S} {sn} bytes")
print(f"published grux-local-2b-r5      manifest sha256:{P} {pn} bytes")
print(f"layers: {len(s['layers'])} scored vs {len(p['layers'])} published")
print(f"config identical: {s['config'] == p['config']} {p['config']['digest']} {p['config']['size']}")
print(f"published layers[:{len(s['layers'])}] identical to scored layers, in order: {p['layers'][:len(s['layers'])] == s['layers']}")
for l in p["layers"][len(s["layers"]):]:
    print(f"  appended: {l['mediaType']} {l['digest']} {l['size']} bytes")
keys = sorted(set(s) | set(p) - {"layers"})
print("non-layer keys identical:", {k: s.get(k) == p.get(k) for k in keys if k != "layers"})
for l in s["layers"]:
    print(f"  scored layer: {l['mediaType']} {l['digest']} {l['size']} bytes")
print(f"replaced 2b (601cd23c66ef) license layer: {[l['digest'] for l in v['layers'] if l['mediaType'].endswith('license')]}")
print(f"scored license layers: {[l['digest'] for l in s['layers'] if l['mediaType'].endswith('license')]}")
lic = [l for l in p["layers"] if l["mediaType"].endswith("license")][0]
raw = open(f"{B}/sha256-{lic['digest'][7:]}", "rb").read()
print(f"license blob hashes to its name: {hashlib.sha256(raw).hexdigest() == lic['digest'][7:]}, {len(raw)} bytes")
