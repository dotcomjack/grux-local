"""A box whose /api/tags lists no matching entry must still score the model in ONE box (regression: 664a8f2 raised
TypeError on the None digest, caught as a lost box, three boxes per model and no rows). --no-registry also fails the
registry fallback, so the digest is recorded as None. No network box, no spend: the box and the meter are fakes."""
import os, sys, tempfile
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)
import common, score, gpu_box
common.reserve = lambda hours, gpu: "rid-test"
common.settle = lambda rid, hours: "0.00"

class FakeBox:
    def __init__(self, *a, **k): self.info = {"id": "fake-box", "gpu_type": "H200", "spot": True}
    def __enter__(self): return self
    def __exit__(self, *e): pass
    def get(self, path, timeout=30): return {"version": common.OLLAMA_PIN} if path == "/api/version" else {"models": []}
    def post(self, path, body, timeout=600): return {}
    def pull(self, model): return "success", 0.1
    def evicted(self): return False

def fake_score_on_box(box, model, run_dir, logp, deadline):
    import json
    for s in score.SETS:
        with open(os.path.join(run_dir, f"{s}.jsonl"), "w") as f:
            for c in score.set_fns(s)[0]():
                f.write(json.dumps({"case": c["id"], "family": c["family"], "status": "scored", "pass": True, "s": 1.0}) + "\n")
    return "done"

if "--no-registry" in sys.argv:
    import watch
    watch.registry_build = lambda *a: (_ for _ in ()).throw(OSError("planted registry failure"))
common.ensure_grux_src()
gpu_box.OllamaBox = FakeBox
score.score_on_box = fake_score_on_box
d = tempfile.mkdtemp(prefix="2099-01-01-test_")
common.RESULTS = tempfile.mkdtemp()
r = score.run_one("qwen3.5:4b", ["H200"], d, 1.2, 60)
print("complete", r["complete"], "digest", r["digest"], "boxes", [b["outcome"] for b in r["boxes"]])
sys.exit(0 if r["complete"] and len(r["boxes"]) == 1 else 1)
