#!/usr/bin/env python3
"""Weekly watch: new open-weight bases at Mac sizes, from the Hugging Face API and the Ollama library.

Hard filters, each logged with its reason when it drops a model:
  - license: must be on base.json's allowlist (Apache-2.0, MIT) or named in license_overrides with a reason;
  - HF-gated weights are out (no token ever goes in a box);
  - size: 1B to 10.5B parameters, and the build must fit its tier (8, 16 or 24 GB Mac);
  - Ollama builds must carry the `tools` badge (Grux calls tools) and not be cloud-only;
  - Hugging Face repos must be a chat model (text-generation or image-text-to-text with a chat template), not a
    quantized copy, a pretrain-only base or another kind of model.
A candidate is scored as the Mac runs it: the default Ollama library build of that size. A Hugging Face base with
no Ollama library build yet waits (rechecked every week). The seen state is keyed by (source, id, digest) and
records when each build was first seen; a build stays new until a complete, fully scored result for its digest
exists in results/autopilot/, so a base re-published under the same name is new again and a failed or refused
week is retried the next.

  python3 recipe/autopilot/watch.py [--out results/autopilot/watch-<date>.json]   live watch
  python3 recipe/autopilot/watch.py --self-test                                   planted red cases
"""
import argparse
import datetime as dt
import glob
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

HF_API = "https://huggingface.co/api/models"
OLLAMA_LIB = "https://ollama.com/library?sort=newest"
REGISTRY = "https://registry.ollama.ai/v2/library"
CHAT_PIPELINES = {"text-generation", "image-text-to-text"}
QUANT_NAME = re.compile(r"(?i)(gguf|awq|gptq|fp8|nvfp4|mxfp|int4|int8|bnb|mlx|onnx|exl2|-q\d)")
STATE_PATH = os.path.join(common.STATE, "watch-state.json")
HF_LIMIT = 500      # newest first; the run reports an error when the oldest of a full page is still inside the window


def scored_digests():
    """Builds with a complete, fully scored result in results/autopilot/: the only thing that makes a build not new,
    so a week whose scoring failed or was refused comes back next week."""
    out = set()
    for p in glob.glob(os.path.join(common.RESULTS, "20*.json")):
        try:
            r = json.load(open(p))
        except ValueError:
            continue
        if r.get("complete") and r.get("digest") and not any(v.get("unscored") for v in r.get("sets", {}).values()):
            out.add(r["digest"])
    return out


def fetch(url, timeout=30, accept=None):
    req = urllib.request.Request(url, headers={"User-Agent": "grux-local-autopilot/1",
                                               **({"Accept": accept} if accept else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# --------------------------------------------------------------------------- license

def classify_license_text(text):
    """('apache-2.0' | 'mit' | None, the first line). Restricted variants (use policies, RAIL) are never Apache or MIT."""
    head = text[:4000]
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")[:120]
    if re.search(r"(?i)acceptable use policy|use restrictions|responsible ai license|\bRAIL\b|community license", head):
        return None, first
    if re.search(r"Apache License", head) and re.search(r"Version 2\.0", head) and \
            re.search(r"TERMS AND CONDITIONS FOR USE, REPRODUCTION", head):
        return "apache-2.0", first
    if "Permission is hereby granted, free of charge" in head:     # the MIT grant itself, not a mention of MIT
        return "mit", first
    return None, first


def license_ok(source_id, license_id, cfg):
    """(ok, reason). An override names its reason; the allowlist is the default."""
    ov = cfg["license_overrides"].get(source_id)
    if ov:
        return True, f"license {ov['license']} allowed by override: {ov['reason']}"
    lid = (license_id or "").lower()
    if lid in cfg["license_allowlist"]:
        return True, f"license {lid}"
    return False, f"license {license_id or 'none stated'} is not on the allowlist (Apache-2.0, MIT)"


# --------------------------------------------------------------------------- Hugging Face

def family_key(name):
    """'Qwen/Qwen3.5-4B' and 'qwen3.5' both give 'qwen3.5'; 'google/gemma-4-e4b-it' and 'gemma4' give 'gemma4'."""
    n = name.split("/")[-1].lower()
    n = re.split(r"[-_:](?:e?\d+(?:\.\d+)?b)(?![a-z])", n)[0]
    return re.sub(r"[^a-z0-9.]", "", n)


def hf_size_key(name):
    m = re.search(r"(?i)(?<![a-z0-9.])(e?\d+(?:\.\d+)?)b(?![a-z0-9])", name.split("/")[-1].replace("_", "-"))
    return m.group(1).lower() + "b" if m else None


def hf_models(org):
    q = (f"{HF_API}?author={org}&sort=createdAt&direction=-1&limit={HF_LIMIT}&expand[]=gated&expand[]=safetensors"
         "&expand[]=cardData&expand[]=tags&expand[]=createdAt&expand[]=pipeline_tag&expand[]=sha&expand[]=config")
    return json.loads(fetch(q, timeout=60))


def hf_card(repo):
    try:
        return json.loads(fetch(f"{HF_API}/{repo}?expand[]=gated&expand[]=cardData&expand[]=safetensors", timeout=30))
    except (urllib.error.URLError, OSError, ValueError) as ex:
        return {"error": str(ex)[:160]}


def judge_hf(m, cfg, since):
    """(verdict, reason, record). verdict: candidate | filtered | baseline."""
    mid = m["id"]
    created = (m.get("createdAt") or "")[:10]
    params = (m.get("safetensors") or {}).get("total")
    lic = (m.get("cardData") or {}).get("license")
    tags = m.get("tags") or []
    rec = {"source": "hf", "id": mid, "sha": m.get("sha"), "created": created, "license": lic,
           "params_b": round(params / 1e9, 2) if params else None, "pipeline": m.get("pipeline_tag"),
           "gated": m.get("gated")}
    if created and created < since:
        return "baseline", f"published {created}, before the {since} field sweep", rec
    if m.get("gated"):
        return "filtered", f"HF-gated weights (gated={m.get('gated')}): no token goes in a box", rec
    if m.get("pipeline_tag") not in CHAT_PIPELINES:
        return "filtered", f"not a chat model (pipeline {m.get('pipeline_tag')})", rec
    if any(t.startswith("base_model:quantized:") for t in tags) or QUANT_NAME.search(mid.split("/")[-1]):
        return "filtered", "a quantized copy of another repo, not a new base", rec
    chat_template = ((m.get("config") or {}).get("tokenizer_config") or {}).get("chat_template")
    if not chat_template:
        return "filtered", "no chat template (pretrain-only or not a chat model)", rec
    ok, why = license_ok("hf:" + mid, lic, cfg)
    if not ok:
        return "filtered", why, rec
    if params is None:
        return "filtered", "size unknown (no safetensors parameter count)", rec
    tier = common.tier_for(params / 1e9)
    if tier is None:
        return "filtered", f"{params / 1e9:.2f}B parameters, outside 1B to 10.5B", rec
    rec["tier"] = tier
    return "candidate", f"{why}, {params / 1e9:.2f}B, tier {tier}", rec


# --------------------------------------------------------------------------- Ollama library

def parse_library(page):
    """Each library entry on the listing: name, capability badges, size badges, pulls, updated (UTC)."""
    out = []
    for block in re.split(r'<li\s', page)[1:]:
        m = re.search(r'href="/library/([^"/?#]+)"', block)
        if not m:
            continue
        caps = re.findall(r'bg-indigo-50[^>]*>\s*([a-z]+)\s*<', block)
        sizes = re.findall(r'bg-\[#ddf4ff\][^>]*>\s*([0-9a-z.\-]+)\s*<', block)
        cloud = bool(re.search(r'>\s*cloud\s*<', block))
        upd = re.search(r'title="([A-Z][a-z]{2} \d{1,2}, \d{4} \d{1,2}:\d{2} [AP]M) UTC"', block)
        pulls = re.search(r'<span[^>]*>([0-9.]+[KMB]?)</span>\s*<span[^>]*>&nbsp;Pulls', block)
        desc = re.search(r'<p class="max-w-lg[^"]*"[^>]*>(.*?)</p>', block, re.S)
        updated = dt.datetime.strptime(upd.group(1), "%b %d, %Y %I:%M %p").date().isoformat() if upd else None
        out.append({"name": m.group(1), "caps": caps, "sizes": sizes, "cloud": cloud, "updated": updated,
                    "pulls": pulls.group(1) if pulls else None,
                    "description": html.unescape(desc.group(1).strip())[:160] if desc else ""})
    return out


def pulls_number(p):
    if not p:
        return 0
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(p[-1], 1)
    return float(p.rstrip("KMB")) * mult


def registry_build(name, size):
    """The default library build of name:size as any client without a key sees it: digest, bytes, license text."""
    raw = fetch(f"{REGISTRY}/{name}/manifests/{size}", accept="application/vnd.docker.distribution.manifest.v2+json")
    import hashlib
    man = json.loads(raw)
    layers = man.get("layers", [])
    model_bytes = sum(l["size"] for l in layers if l["mediaType"].endswith((".model", ".projector")))
    lic_layer = next((l for l in layers if l["mediaType"].endswith(".license")), None)
    text = fetch(f"{REGISTRY}/{name}/blobs/{lic_layer['digest']}", timeout=60).decode("utf-8", "replace") \
        if lic_layer else ""
    return {"digest": "sha256:" + hashlib.sha256(raw).hexdigest(), "model_bytes": model_bytes,
            "format": man.get("format"), "runner": man.get("runner"), "license_text": text}


def judge_ollama_entry(e, cfg, since, base_family):
    """Listing-level filters (no network). Returns (verdict, reason, sizes to look at)."""
    if family_key(e["name"]) == family_key(base_family):
        return "baseline", f"the current base family ({base_family})", []
    if e["updated"] and e["updated"] < since:
        return "baseline", f"last updated {e['updated']}, before the {since} field sweep", []
    if "embedding" in e["caps"]:
        return "filtered", "an embedding model", []
    if e["cloud"] and not e["sizes"]:
        return "filtered", "cloud-only (no local build)", []
    if "tools" not in e["caps"]:
        return "filtered", f"no tools badge (Grux calls tools); badges {e['caps'] or 'none'}", []
    sizes = [s for s in e["sizes"] if common.tier_for(common.params_from_tag(s)) is not None]
    if not sizes:
        return "filtered", f"sizes {e['sizes'] or 'none listed'}, none in 1B to 10.5B", []
    return "check", "", sizes


def judge_ollama_build(name, size, build, cfg):
    """Build-level filters on what the registry serves: license text, tier fit."""
    lic, first = classify_license_text(build["license_text"]) if build["license_text"] else (None, "")
    rec = {"source": "ollama", "id": f"{name}:{size}", "digest": build["digest"], "format": build["format"],
           "runner": build["runner"], "model_gb": round(build["model_bytes"] / 1e9, 2), "license": lic,
           "license_first_line": first, "params_b": common.params_from_tag(size)}
    if not build["license_text"]:
        return "filtered", "no license file in the build", rec
    if lic is None:
        # Some builds ship a pointer, not the text: "MIT (declared by X in the pinned source model card)" and the
        # Hugging Face URL. Read that card; the HF filters (gated, license) then decide, never the pointer itself.
        ref = re.search(r"huggingface\.co/([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)", build["license_text"])
        if ref:
            card = build.get("hf_card") or hf_card(ref.group(1))
            rec["hf_id"] = ref.group(1)
            if card.get("gated"):
                return "filtered", f"its source weights {ref.group(1)} are HF-gated (gated={card['gated']})", rec
            lic = (card.get("cardData") or {}).get("license")
            rec["license"] = lic
            rec["license_source"] = f"the build's license file points to {ref.group(1)}, whose model card declares {lic}"
    ok, why = license_ok(f"ollama:{name}:{size}", lic or first, cfg)
    if not ok:
        return "filtered", why, rec
    tier = common.tier_for(rec["params_b"])
    fits, est, budget = common.fits_tier(build["model_bytes"], tier)
    rec.update({"tier": tier, "est_loaded_gb": est, "tier_budget_gb": budget})
    if not fits:
        return "filtered", f"about {est} GB loaded, over the {budget} GB a {tier} tier Mac can give a model", rec
    return "candidate", f"{why}, {size}, tier {tier}, about {est} GB loaded", rec


# --------------------------------------------------------------------------- the run

def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"seen": {}}


def pick(cands, headline):
    """The single most promising new candidate, by a fixed rule: the headline tier first, newest, then pulls."""
    pool = [c for c in cands if c.get("scorable")]
    pool.sort(key=lambda c: (c["tier"] != headline, -(int((c.get("updated") or "0000-00-00").replace("-", ""))),
                             -c.get("pulls_n", 0)))
    return pool[0]["id"] if pool else None


def run(out_path, use_state=True):
    cfg = common.base_config()
    since = cfg["since"]
    state = load_state() if use_state else {"seen": {}}
    report = {"date": common.today(), "since": since, "rule": __doc__.strip().splitlines()[0],
              "candidates": [], "waiting": [], "filtered": [], "baseline": [], "errors": []}

    # Ollama library: the builds a Mac can pull.
    page = fetch(OLLAMA_LIB, timeout=60).decode("utf-8", "replace")
    entries = parse_library(page)
    report["ollama_entries"] = len(entries)
    if not entries:
        report["errors"].append("the Ollama library listing parsed to 0 entries (page layout changed?)")
    ollama_by_key = {}
    for e in entries:
        verdict, reason, sizes = judge_ollama_entry(e, cfg, since, cfg["family"])
        if verdict == "baseline":
            report["baseline"].append(["ollama", e["name"], reason])
            continue
        if verdict == "filtered":
            report["filtered"].append({"source": "ollama", "id": e["name"], "reason": reason,
                                       "updated": e["updated"], "badges": e["caps"], "sizes": e["sizes"]})
            continue
        for size in sizes:
            try:
                build = registry_build(e["name"], size)
            except (urllib.error.URLError, OSError, ValueError, KeyError) as ex:
                report["errors"].append(f"{e['name']}:{size}: registry read failed: {str(ex)[:160]}")
                continue
            v, why, rec = judge_ollama_build(e["name"], size, build, cfg)
            rec.update({"updated": e["updated"], "pulls": e["pulls"], "pulls_n": pulls_number(e["pulls"]),
                        "description": e["description"], "reason": why})
            if v == "filtered":
                report["filtered"].append(rec)
                continue
            ollama_by_key[(family_key(e["name"]), size.lower())] = rec

    # Hugging Face: new bases from the watched orgs; linked to an Ollama build when one exists.
    hf_cands, gated_keys = [], {}
    for org in cfg["hf_orgs"]:
        try:
            models = hf_models(org)
        except (urllib.error.URLError, OSError, ValueError) as ex:
            report["errors"].append(f"HF {org}: {str(ex)[:160]}")
            continue
        if len(models) >= HF_LIMIT and (models[-1].get("createdAt") or "")[:10] >= since:
            report["errors"].append(f"HF {org}: {HF_LIMIT} repos all inside the window; older new bases may be unread")
        for m in models:
            if m.get("gated"):
                gated_keys[(family_key(m["id"]), hf_size_key(m["id"]))] = m["id"]
            v, why, rec = judge_hf(m, cfg, since)
            rec["reason"] = why
            if v == "baseline":
                report["baseline"].append(["hf", m["id"], why])
            elif v == "filtered":
                report["filtered"].append(rec)
            else:
                hf_cands.append(rec)
    for rec in hf_cands:
        key = (family_key(rec["id"]), hf_size_key(rec["id"]))
        o = ollama_by_key.get(key)
        if o:
            o["hf_id"] = rec["id"]
        else:
            report["waiting"].append({**rec, "reason": rec["reason"] + "; no Ollama library build yet (rechecked weekly)"})

    done = scored_digests()
    for key, rec in list(ollama_by_key.items()):
        if key in gated_keys:     # the same base's weights are HF-gated: the recipe could never train it
            rec["reason"] = f"its Hugging Face weights {gated_keys[key]} are HF-gated: no token goes in a box"
            report["filtered"].append(rec)
            continue
        rec["first_seen"] = state["seen"].get(f"ollama|{rec['id']}|{rec['digest']}", {}).get("first_seen", report["date"])
        rec["new"] = rec["digest"] not in done
        rec["scorable"] = True
        report["candidates"].append(rec)
    report["pick"] = pick([c for c in report["candidates"] if c["new"]], cfg["headline_tier"])
    report["pick_rule"] = "headline tier (4b) first, then newest update, then pulls; only builds not yet scored"
    report["counts"] = {"candidates": len(report["candidates"]),
                        "new_candidates": sum(c["new"] for c in report["candidates"]),
                        "waiting": len(report["waiting"]), "filtered": len(report["filtered"]),
                        "baseline": len(report["baseline"]), "errors": len(report["errors"])}
    common.write_json(out_path, report)
    if use_state:
        for c in report["candidates"]:
            state["seen"].setdefault(f"ollama|{c['id']}|{c['digest']}", {"first_seen": report["date"]})
        for w in report["waiting"]:
            state["seen"].setdefault(f"hf|{w['id']}|{w['sha']}", {"first_seen": report["date"], "waiting": True})
        common.write_json(STATE_PATH, state)
    return report


# --------------------------------------------------------------------------- planted red cases

def self_test():
    cfg = common.base_config()
    since = cfg["since"]
    tmpl = "{%- for m in messages %}{{ m.content }}{% endfor %}"
    clean = {"id": "planted/Clean-4B-Instruct", "createdAt": "2026-11-01T00:00:00Z", "gated": False,
             "safetensors": {"total": 4_200_000_000}, "cardData": {"license": "apache-2.0"},
             "tags": ["license:apache-2.0"], "pipeline_tag": "text-generation", "sha": "a",
             "config": {"tokenizer_config": {"chat_template": tmpl}}}
    gated = {**clean, "id": "planted/Gated-4B-Instruct", "gated": "manual"}
    other = {**clean, "id": "planted/OtherLicense-4B-Instruct", "cardData": {"license": "other"}}
    results = {m["id"]: judge_hf(m, cfg, since)[:2] for m in (clean, gated, other)}
    ollama_clean = {"license_text": "Apache License\nVersion 2.0, January 2004\n\nTERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n...", "digest": "sha256:x",
                    "model_bytes": 3_300_000_000, "format": "gguf", "runner": "llamacpp"}
    ollama_llama = {**ollama_clean, "license_text": "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT\nLlama 3.2 Version..."}
    ollama_custom = {**ollama_clean, "license_text": "Planted Model License\nYou may use these weights only..."}
    ollama_mentions_mit = {**ollama_clean, "license_text": "Planted Terms\nThe tokenizer is under the MIT License; "
                                                          "the weights may not be redistributed."}
    results["ollama planted-clean:4b"] = judge_ollama_build("planted-clean", "4b", ollama_clean, cfg)[:2]
    results["ollama planted-llama:3b"] = judge_ollama_build("planted-llama", "3b", ollama_llama, cfg)[:2]
    results["ollama planted-custom:4b"] = judge_ollama_build("planted-custom", "4b", ollama_custom, cfg)[:2]
    results["ollama planted-mentions-mit:4b"] = judge_ollama_build("planted-mentions-mit", "4b", ollama_mentions_mit,
                                                                   cfg)[:2]
    pointer = "MIT (declared in the pinned source model card)\nhttps://huggingface.co/planted/Pointer-4B/tree/abc\n"
    results["ollama planted-pointer-gated:4b"] = judge_ollama_build("planted-pointer-gated", "4b", {
        **ollama_clean, "license_text": pointer, "hf_card": {"gated": "auto", "cardData": {"license": "mit"}}}, cfg)[:2]
    results["ollama planted-pointer-mit:4b"] = judge_ollama_build("planted-pointer-mit", "4b", {
        **ollama_clean, "license_text": pointer, "hf_card": {"gated": False, "cardData": {"license": "mit"}}}, cfg)[:2]
    for k, (v, why) in results.items():
        print(f"{v:9} {k}: {why}")
    want = {"planted/Clean-4B-Instruct": "candidate", "planted/Gated-4B-Instruct": "filtered",
            "planted/OtherLicense-4B-Instruct": "filtered", "ollama planted-clean:4b": "candidate",
            "ollama planted-llama:3b": "filtered", "ollama planted-custom:4b": "filtered",
            "ollama planted-pointer-gated:4b": "filtered", "ollama planted-mentions-mit:4b": "filtered", "ollama planted-pointer-mit:4b": "candidate"}
    bad = [k for k, v in want.items() if results[k][0] != v]
    if "gated" not in results["planted/Gated-4B-Instruct"][1].lower():
        bad.append("gated model filtered for the wrong reason")
    if bad:
        print("SELF-TEST FAIL:", bad)
        sys.exit(1)
    print("SELF-TEST PASS: gated and non-allowlisted licenses are filtered; the clean twins pass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-state", action="store_true", help="do not read or write the seen-state file")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    out = a.out or os.path.join(common.RESULTS, f"watch-{common.today()}.json")
    r = run(out, use_state=not a.no_state)
    print(json.dumps(r["counts"]))
    for c in r["candidates"]:
        print(f"CANDIDATE {c['id']} tier {c['tier']} {c['license']} updated {c['updated']} pulls {c['pulls']} "
              f"{'new' if c['new'] else 'seen'} hf {c.get('hf_id')}")
    for w in r["waiting"]:
        print(f"WAITING   {w['id']}: {w['reason']}")
    print(f"PICK {r['pick']}")
    for e in r["errors"]:
        print("ERROR", e)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
