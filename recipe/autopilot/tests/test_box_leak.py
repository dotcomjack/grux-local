"""A box whose start fails AFTER its Lease exists must be deleted and its Lease stopped (OllamaBox and TrainBox)."""
import os, sys, types
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R)
import lease, gpu_box, train_box

class FakeSB:
    def __init__(self, d): self.id, self.d = "fake-sb-1", d
    def set_ttl(self, m): pass
    def model_dump(self): return {"id": self.id, "gpu_type": "H200"}
    def get_preview_link(self, port): raise RuntimeError("planted: preview link failed after the lease")
    process = types.SimpleNamespace(create_session=lambda n: None, execute_session_command=lambda *a, **k: None,
                                    exec=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("planted: exec failed after the lease")))
class FakeD:
    def __init__(self): self.live, self.deleted = set(), []
    def create(self, params, timeout=0): sb = FakeSB(self); self.live.add(sb.id); return sb
    def delete(self, sb, wait=False, timeout=0): self.deleted.append(sb.id); self.live.discard(sb.id)
    def list(self, q=None): return [s for s in self.live]

class NoLeaseSB(FakeSB):
    def set_ttl(self, m): raise RuntimeError("planted: set_ttl failed, the Lease never forms")
class NoLeaseD(FakeD):
    def create(self, params, timeout=0): sb = NoLeaseSB(self); self.live.add(sb.id); return sb
class NoCreateD(FakeD):
    def create(self, params, timeout=0): raise RuntimeError("planted: no spot capacity")

fails = 0
cases = [(f"{cls} {why}", cls, d) for why, d in (("start fails after the Lease", FakeD),
                                                    ("Lease never forms", NoLeaseD),
                                                    ("create fails", NoCreateD))
         for cls in ("OllamaBox", "TrainBox")]
for name, cls, dcls in cases:
    make = (lambda: gpu_box.OllamaBox(["H200"], "t", 10, owner="grux-local")) if cls == "OllamaBox" else \
        (lambda: train_box.TrainBox(["H200"], "t", 10, owner="grux-local"))
    gpu_box.Daytona = train_box.Daytona = dcls
    box = make()
    d = box.d
    try:
        with box:
            pass
        print(name, "start did not fail?")
        fails += 1
    except Exception as e:
        if "planted" not in str(e):        # the original error must surface, not a cleanup error
            print(f"FAIL {name}: the error that surfaced is not the planted one: {e}")
            fails += 1
    live_leases = list(lease._live)
    ok = not d.live and not live_leases
    print(f"{'PASS' if ok else 'FAIL'} {name}: boxes left {sorted(d.live)}, leases still beating {live_leases}")
    fails += not ok
    lease._live.clear()
sys.exit(1 if fails else 0)
