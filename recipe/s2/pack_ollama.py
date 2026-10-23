"""pack_ollama.py <name> <out.tar>: one Ollama model (v2 manifest link, manifest blob, config, layers) as a tar
rooted at ~/.ollama/models, so `tar xf` there makes `ollama list` show it on another machine."""
import json, os, sys, tarfile
root = "/root/.ollama/models"
name, out = sys.argv[1], sys.argv[2]
link = f"manifests-v2/ollama.com/library/{name}/latest"
target = os.path.realpath(os.path.join(root, link))
m = json.load(open(target))
blobs = {os.path.relpath(target, root)} | {"blobs/" + d["digest"].replace(":", "-") for d in m["layers"] + [m["config"]]}
with tarfile.open(out, "w") as t:
    t.add(os.path.join(root, link), arcname=link)          # the symlink itself
    for b in sorted(blobs):
        t.add(os.path.join(root, b), arcname=b)
print("packed", len(blobs), "blobs", round(os.path.getsize(out) / 1e9, 2), "GB")
