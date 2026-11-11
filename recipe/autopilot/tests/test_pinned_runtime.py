"""Every scoring box runs the pinned Ollama image, and a box reporting another runtime scores nothing (2026-11-11:
with :latest, one new model scored on a newer runtime would lead the page alone). No network box, no spend."""
import os, sys, tempfile
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)
import common, score, gpu_box
common.reserve = lambda hours, gpu: "rid-test"
common.settle = lambda rid, hours: "0.00"
seen = {}

def make_box(version):
    class FakeBox:
        def __init__(self, *a, **k):
            seen["image"] = k.get("image")
            self.info = {"id": "fake-box", "gpu_type": "H200", "spot": True}
        def __enter__(self): return self
        def __exit__(self, *e): pass
        def get(self, path, timeout=30): return {"version": version} if path == "/api/version" else {"models": []}
        def post(self, path, body, timeout=600): return {}
        def pull(self, model): return "success", 0.1
        def evicted(self): return False
    return FakeBox

scored = []
score.score_on_box = lambda *a: scored.append(1) or "done"
common.ensure_grux_src()
common.RESULTS = tempfile.mkdtemp()
fails = []
assert common.OLLAMA_IMAGE == "ollama/ollama:" + common.OLLAMA_PIN and "latest" not in common.OLLAMA_IMAGE

gpu_box.OllamaBox = score.OllamaBox = make_box("0.41.0")
r = score.run_one("qwen3.5:4b", ["H200"], tempfile.mkdtemp(prefix="2099-01-01-a_"), 1.2, 60)
if seen.get("image") != common.OLLAMA_IMAGE: fails.append(f"box image {seen.get('image')!r}, want {common.OLLAMA_IMAGE}")
if scored: fails.append("a box on runtime 0.41.0 was scored")
if len(r["boxes"]) != 1 or "not the pinned" not in r["boxes"][0]["outcome"]:
    fails.append(f"want one refused box, got {[b.get('outcome') for b in r['boxes']]}")
if r["complete"]: fails.append("a refused runtime reported complete")

print("FAIL: " + "; ".join(fails) if fails else "PASS: pinned image passed to the box; runtime 0.41.0 refused, nothing scored, one box")
sys.exit(1 if fails else 0)
