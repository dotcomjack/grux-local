"""train_run ends "done" when its scores are in, even if the model's pack was skipped for disk or failed to download
(pull.json says which). The autopilot must then refuse to write a Mini gate request, naming the reason, because the
gate's first step untars that pack. No box, no network: run dirs are written here."""
import json, os, sys, tempfile
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)
import train

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    fails += not ok


d = tempfile.mkdtemp()
json.dump({"model": "grux-m", "pull": "skipped: needs 22.00 GB plus a 10 GB floor, 13.00 GB free"},
          open(os.path.join(d, "pull.json"), "w"))
why = train.pack_refusal(d, "grux-m")
check("a pack skipped for disk refuses the gate request, naming the reason",
      why is not None and why.startswith("scores stand; pack not pulled: skipped: needs 22.00 GB"), str(why))
os.remove(os.path.join(d, "pull.json"))
check("no pull.json refuses too", "no pull.json" in (train.pack_refusal(d, "grux-m") or ""))
open(os.path.join(d, "ollama-grux-m.tar"), "wb").write(b"x")
check("a pack that is here passes", train.pack_refusal(d, "grux-m") is None)
src = open(train.__file__).read()
check("main refuses on pack_refusal right after the status check and before the gate request is written",
      -1 < src.find('refuse(f"train_run.py ended') < src.find("why = pack_refusal(run_dir, tag)") < src.find("common.write_json(gpath, gate)"))
print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
