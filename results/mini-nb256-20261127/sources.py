"""Run on the Mac mini: prints, for each raw file results/v1.json's 2B (manifest 86255f20) is built from, its sha256,
size and the time it was last written (US Eastern, the Mini's clock), keyed by its path in this folder, so
results/build_v1.py can verify the copies here and date each run without trusting a checkout's file times.

  python3 sources.py > SOURCES.json

The four blobs are the two manifests (86255f20 and the 1622ac1b it replaced) and their params layers, read from
Ollama 0.40's store, where a manifest is a blob too. probe/batch-extract.log is the lines of the server log that
probe/batch_test.sh wrote to /tmp/batch.log that name the batch size, a restored checkpoint, a prompt's evaluation or a
chat request: grep -E "n_batch +=|n_ubatch +=|restored context checkpoint|prompt eval time|POST +\"/api/chat\"".
"""
import datetime, hashlib, json, os
H = os.path.expanduser("~")
E, B = f"{H}/grux-eval", f"{H}/.ollama/models/blobs"
FILES = {}
for s in ("-holdout", "-holdout2", ""):
    for x in (".jsonl", ".meta.json"):
        FILES[f"runs/nb256/grux-local-2b-r5-nb256{s}{x}"] = f"{E}/runs/nb256/grux-local-2b-r5-nb256{s}{x}"
        FILES[f"runs/nb128/grux-local-2b-r5-nb128{s}{x}"] = f"{E}/runs/nb128/grux-local-2b-r5-nb128{s}{x}"
for x in (".jsonl", ".meta.json"):
    FILES[f"runs/nb128/grux-local-2b-r5-nb128-local{x}"] = f"{E}/runs/nb128/grux-local-2b-r5-nb128-local{x}"
    for d in ("nb256-8k", "nb128-8k"):
        n = d.split("-")[0]
        FILES[f"runs/{d}/grux-local-2b-r5-{n}-local{x}"] = f"{E}/runs/{d}/grux-local-2b-r5-{n}-local{x}"
    for s in ("-holdout", "-local"):
        FILES[f"runs/nb256-published/grux-local-2b-r5-nb256{s}{x}"] = f"{E}/runs/nb256-published/grux-local-2b-r5-nb256{s}{x}"
    FILES[f"runs/r5-pass1/grux-local-2b-r5pilot{x}"] = f"{E}/runs/r5-pass1/grux-local-2b-r5pilot{x}"
for n in ("ps-fresh.json", "ps-after.json", "footprint-after.txt"):
    FILES[f"runs/nb256-published/{n}"] = f"{E}/runs/nb256-published/{n}"
for m in ("nb256", "nb128"):
    FILES[f"fname-ab/fn8k-grux-local-2b-r5-{m}.txt"] = f"{E}/fname-ab/fn8k-grux-local-2b-r5-{m}.txt"
for h in ("86255f201bd61aef2d4b4d43b00de50f6f491195f55e474e041253bdd9818a99",
          "1622ac1b80afb8841fc588fbc57eff60284f318d021311322a84df4aa097251e",
          "703293fc57f488e92240b59a4df4e2915999d1ce16189cc2dcc4e389fea89969",
          "9371364b27a52acac9d87f88bd93c9db1174d8d6ec57f6888925cdc1788871ff"):
    FILES[f"blobs/sha256-{h}"] = f"{B}/sha256-{h}"
FILES.update({"Modelfile256": f"{E}/nb128/Modelfile256", "Modelfile128": f"{E}/nb128/Modelfile",
              "show.json": f"{E}/nb256-show.json",
              "probe/batch-extract.log": f"{E}/nb256-probe-extract.log", "probe/batch_probe.py": f"{E}/batch_probe.py",
              "probe/batch_test.sh": f"{E}/batch_test.sh",
              "scripts/run-nb128.sh": f"{E}/run-nb128.sh", "scripts/run-nb256.sh": f"{E}/run-nb256.sh",
              "scripts/run-nb256b.sh": f"{E}/run-nb256b.sh", "scripts/measure-nb256.sh": f"{E}/measure-nb256.sh",
              "scripts/run-nb128.log": f"{E}/run-nb128.log", "scripts/run-nb256.log": f"{E}/run-nb256.log",
              "scripts/measure-nb256.log": f"{E}/measure-nb256.log"})
if __name__ == "__main__":
    out = {}
    for rel, f in FILES.items():
        b = open(f, "rb").read()
        out[rel] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b),
                    "written": datetime.datetime.fromtimestamp(os.path.getmtime(f)).isoformat(timespec="seconds")}
    print(json.dumps(out, indent=1))
