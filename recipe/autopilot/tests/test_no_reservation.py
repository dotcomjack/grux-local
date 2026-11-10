"""No reservation, no box: OllamaBox and TrainBox raise before any create without a reservation id (argument or
GRUX_METER_RID). Pass another recipe folder as argv[1] to run it against an older copy (it fails there)."""
import os, sys, types
R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R)
import gpu_box, train_box
os.environ.pop("GRUX_METER_RID", None)
created = []
class SB:
    id = "x"
    def set_ttl(self, m): pass
    def model_dump(self): return {"id": "x"}
    def get_preview_link(self, p): raise RuntimeError("stop here")
    process = types.SimpleNamespace(create_session=lambda n: None, execute_session_command=lambda *a, **k: None,
                                    exec=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop here")))
class D:
    def create(self, p, timeout=0): created.append(1); return SB()
    def delete(self, sb, wait=False, timeout=0): pass
    def list(self, q=None): return []
gpu_box.Daytona = train_box.Daytona = D
for make in (lambda: gpu_box.OllamaBox(["H200"], "t", 10), lambda: train_box.TrainBox(["H200"], "t", 10)):
    try:
        with make(): pass
    except RuntimeError as e:
        print("  raised:", str(e)[:70])
print("boxes created without a reservation:", len(created))
sys.exit(1 if created else 0)
