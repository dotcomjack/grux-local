#!/usr/bin/env python3
"""Numbers check: every surface that states a Grux Local number must state the one in results/v1.json.

  python3 results/check.py                                  # the live surfaces and the live registry
  python3 results/check.py --card README.md --readme ../grux/README.md --site page.html --registry https://<host>

Each surface is a path or a URL; the blog is the dotcomjack.com announcement, checked on its prose. Two checks per surface, both from v1.json:
  rows   every table row and headline sentence that surface carries, matched whole, so a right number in
         the wrong row, or a row that went missing, fails;
  shape  every number on the surface shaped like one of ours ("N (P%)", "N of 464", "N out of 70",
         "X.XX GB", "X.X s") must be one of ours, so a stray wrong number elsewhere on the page fails too.
  scope  a before-training number measured on another Mac (the 9B's, on a MacBook Pro) must name that Mac
         near the size on every surface, so it never reads as the Mac mini's;
  claims an absolute behaviour claim ("it ignores instructions hidden in...", "are not followed",
         "it says so instead of pretending") fails while v1.json shows that family below 100%: no
         tag is perfect at it, so the surface must say "trained to" and let the table carry the rate.
The registry check ties the numbers to the files: each tag's manifest must hash to v1.json's digest.
--blobs N fires N ranged GETs at once across every blob of every tag, the way `ollama pull` does, and
fails if more than 5% do not answer 206: on 2026-10-23 the preview passed every check while production
dropped about 1 blob request in 10 at the upstream, which only a burst of real ranged GETs showed.
Exit 0 only when every surface passes; a surface with nothing matched is a FAIL, never a pass.
"""
import argparse
import concurrent.futures
import hashlib
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = {
    "card": "https://raw.githubusercontent.com/dotcomjack/grux-local/main/README.md",
    "readme": "https://raw.githubusercontent.com/dotcomjack/grux/main/README.md",
    "site": "https://gruxai.com/answers/grux-local-model/",
    "blog": "https://dotcomjack.com/blog/jybp4hg1",
    "registry": "https://gruxai.com",
}
UA = {"User-Agent": "grux-local-numbers-check"}


def fetch(src, headers=None):
    if re.match(r"https?://", src):
        req = urllib.request.Request(src, headers={**UA, **(headers or {})})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read(), dict(r.headers)
    return open(src, "rb").read(), {}


def text_of(raw, src):
    s = raw.decode("utf-8", "replace")
    if "<html" in s[:2000].lower():
        m = re.search(r"<main\b.*?</main>", s, re.S | re.I)
        s = m.group(0) if m else s
        s = re.sub(r"<(script|style)\b.*?</\1>", " ", s, flags=re.S | re.I)
        s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    else:
        s = re.sub(r"^\s*>\s?", "", s, flags=re.M)
        s = s.replace("**", "").replace("`", "").replace("|", " ")
    return re.sub(r"\s+", " ", s)


def readme_scope(md):
    """The Grux README talks about much else; only the announcement and the Grux Local section are ours."""
    callout = re.search(r"^> \*\*New: Grux Local.*?(?=^[^>])", md, re.S | re.M)
    section = re.search(r"^## Grux Local\n.*?(?=^---|^## )", md, re.S | re.M)
    return (callout.group(0) if callout else "") + "\n" + (section.group(0) if section else "")


# Third-person absolutes, per scored family. "Trained to ignore" or an imperative under a "trained to do"
# heading is fine; "it ignores" states a 100% that the table contradicts. Found by the 2026-10-23 copy
# review on the card and the site: three such claims, while the families scored 65 to 118 of 120.
ABSOLUTE = {
    "injection": [r"\bignores instructions hidden\b", r"\b(?:are|is) not followed\b", r"\bnever follows\b"],
    "chat_honesty": [r"\bit says so instead of pretending\b", r"\bsays no honestly\b"],
    "chat_tools": [r"\bit calls the right\b", r"(?<![Tt]rained to )\bCalls the right Grux tool\b"],
}


def absolutes(text, v1):
    out = []
    for fam, pats in ABSOLUTE.items():
        rates = [(t, d["grux_local"]["families"][fam]) for t, d in v1["tags"].items()]
        imperfect = [f"{t} {r['pass']}/{r['of']}" for t, r in rates if r["pass"] < r["of"]]
        if not imperfect:
            continue
        for pat in pats:
            for m in re.finditer(pat, text):
                out.append(f"{m.group(0)!r} while {fam} is {', '.join(imperfect)}")
    return out


def pct(n, of):
    return round(100 * n / of)


def expected(v1):
    t = v1["tags"]
    rows, scope, allowed = [], [], {"pairs": set(), "of_tasks": set(), "of_hidden": set(), "gb": set(), "s": set()}
    for tag, d in t.items():
        T, a, b = d["size"], d["grux_local"], d["before_training"]
        of, hc = a["of"], a["hidden_instruction_cases"]
        ap, bp = pct(a["pass"], of), pct(b["pass"], of)
        rows += [
            ("all", f"{T} all tasks: {b['pass']} ({bp}%) to {a['pass']} ({ap}%)",
             rf"{T}, all (?:{of} )?tasks {b['pass']} \({bp}%\) {a['pass']} \({ap}%\)"),
            ("hidden", f"{T} hidden instructions: {b['hidden_instructions_ignored']} to {a['hidden_instructions_ignored']} of {hc}",
             rf"{T}, hidden instructions ignored {b['hidden_instructions_ignored']} of {hc} {a['hidden_instructions_ignored']} of {hc}"),
            ("loaded", f"{T} loaded {d['loaded_gb']:.2f} GB on {d['made_for']}",
             rf"grux-local:{tag}\b[^0-9]{{0,40}}?{re.escape(d['made_for'])} {d['loaded_gb']:.2f} GB"),
            ("chat", f"{T} chat p95 {a['chat_p95_s']:.1f} s",
             rf"{a['chat_p95_s']:.1f} s(?:econds)?(?: on the {T}| \({T}\))"),
        ]
        other = re.search(r", on an? (.+?) \(", d["before_training_model"])
        if other:
            m = re.escape(other.group(1))
            scope.append(("scope", f"{T} before training named as the {other.group(1)}",
                          rf"\b{T}\b.{{0,200}}?\b{m}\b|\b{m}\b.{{0,200}}?\b{T}\b"))
        allowed["pairs"] |= {(a["pass"], ap), (b["pass"], bp)}
        allowed["of_tasks"] |= {a["pass"], b["pass"]}
        allowed["of_hidden"] |= {a["hidden_instructions_ignored"], b["hidden_instructions_ignored"]}
        allowed["gb"].add(f"{d['loaded_gb']:.2f}")
        allowed["s"].add(f"{a['chat_p95_s']:.1f}")
    big, small, nine = t["4b"], t["2b"], t["9b"]
    of = big["grux_local"]["of"]
    headline = {
        "readme": [
            ("headline", "4B headline", rf"\b{big['grux_local']['pass']} of {of} held-out Grux tasks right, against "
                                        rf"{big['before_training']['pass']} for the same model before training"),
            ("headline", "4B hidden headline", rf"\bin {big['grux_local']['hidden_instructions_ignored']} of "
                                               rf"{big['grux_local']['hidden_instruction_cases']} held-out cases, against "
                                               rf"{big['before_training']['hidden_instructions_ignored']}\b"),
            ("headline", "2B headline", rf"2B, made for 8 GB Macs, went from {small['before_training']['pass']} to "
                                        rf"{small['grux_local']['pass']} of {of}\b"),
            ("headline", "9B headline", rf"9B, made for {re.escape(nine['made_for'])}, went from "
                                        rf"{nine['before_training']['pass']} to {nine['grux_local']['pass']} of {of}\b"),
        ] + scope,
        "site": [
            ("headline", "4B lead", rf"the 4B got {big['grux_local']['pass']} right, against "
                                    rf"{big['before_training']['pass']} for the same model before training"),
        ] + scope,
        # The dotcomjack.com announcement (2026-10-23) carries prose, not the tables.
        "blog": [
            ("headline", "blog task count", rf"across {of} held-out Grux tasks"),
            ("headline", "blog 4B", rf"the 4B version got {big['grux_local']['pass']} right, against "
                                    rf"{big['before_training']['pass']} for the same model before training"),
            ("headline", "blog 2B", rf"2B version, made for 8 GB Macs, went from {small['before_training']['pass']} "
                                    rf"to {small['grux_local']['pass']}\b"),
            ("headline", "blog 9B", rf"9B version, made for {re.escape(nine['made_for'])}, went from "
                                    rf"{nine['before_training']['pass']} to {nine['grux_local']['pass']}\b"),
        ] + scope,
        "card": scope,
    }
    return rows, headline, allowed, of, big["grux_local"]["hidden_instruction_cases"]


def shape(text, allowed, of, hc):
    bad, n = [], 0
    for m in re.finditer(r"\b(\d+) \((\d+)%\)", text):
        n += 1
        if (int(m.group(1)), int(m.group(2))) not in allowed["pairs"]:
            bad.append(m.group(0))
    for m in re.finditer(rf"\b(\d+) of {of}\b", text):
        n += 1
        if int(m.group(1)) not in allowed["of_tasks"]:
            bad.append(m.group(0))
    for m in re.finditer(rf"\b(\d+) (?:times )?(?:out )?of {hc}\b", text):
        n += 1
        if int(m.group(1)) not in allowed["of_hidden"]:
            bad.append(m.group(0))
    for m in re.finditer(r"\b(\d+\.\d+) GB\b", text):
        n += 1
        if m.group(1) not in allowed["gb"]:
            bad.append(m.group(0))
    for m in re.finditer(r"\b(\d+\.\d) s(?:econds)?\b", text):
        n += 1
        if m.group(1) not in allowed["s"]:
            bad.append(m.group(0))
    return n, bad


def blob_burst(registry, v1, n):
    """n ranged GETs at once, spread over every layer of every tag, each at a different offset."""
    layers = []
    for tag, d in v1["tags"].items():
        body, _ = fetch(f"{registry.rstrip('/')}/v2/grux/grux-local/manifests/{tag}",
                        {"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
        layers += [(l["digest"], l["size"]) for l in json.loads(body)["layers"] if l["size"] > 4096]
    layers = sorted(set(layers), key=lambda x: -x[1])
    jobs = []
    for i in range(n):
        digest, size = layers[i % len(layers)]
        start = (i * 7919 * 4096) % max(1, size - 4096)
        jobs.append((digest, start))

    def one(job):
        digest, start = job
        url = f"{registry.rstrip('/')}/v2/grux/grux-local/blobs/{digest}"
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{start + 4095}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status if len(r.read()) == 4096 else -1
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:
            return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 40)) as pool:
        codes = list(pool.map(one, jobs))
    bad = [c for c in codes if c != 206]
    ok = len(bad) <= n * 0.05
    print(f"{'PASS' if ok else 'FAIL'} blobs: {n - len(bad)} of {n} ranged GETs answered 206 across "
          f"{len(layers)} layers" + (f", failures {sorted(bad)}" if bad else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    for k, v in LIVE.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--results", default=os.path.join(HERE, "v1.json"))
    ap.add_argument("--blobs", type=int, default=0, help="ranged blob GETs to fire at once (0 = skip)")
    a = ap.parse_args()
    v1 = json.load(open(a.results))
    rows, headline, allowed, of, hc = expected(v1)
    failed = 0
    for name in ("card", "readme", "site", "blog"):
        src = getattr(a, name)
        try:
            raw, _ = fetch(src)
        except Exception as e:
            print(f"FAIL {name}: could not read {src}: {e}")
            failed += 1
            continue
        md = raw.decode("utf-8", "replace")
        text = text_of(readme_scope(md).encode() if name == "readme" else raw, src)
        want = ([] if name == "blog" else rows) + headline.get(name, [])
        miss = [label for _, label, rx in want if not re.search(rx, text)]
        n, bad = shape(text, allowed, of, hc)
        claims = absolutes(text, v1)
        # Prose (the blog) has no table shapes, so there every row matching is the floor instead.
        ok = not miss and not bad and not claims and (n > 0 or (name == "blog" and bool(want)))
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {len(want) - len(miss)} of {len(want)} rows match, "
              f"{n} numbers checked by shape, {len(bad)} wrong, {len(claims)} absolute claims ({src})")
        for label in miss:
            print(f"     missing or different: {label}")
        for b in bad:
            print(f"     not a v1.json number: {b!r}")
        for c in claims:
            print(f"     absolute claim: {c}")
    for tag, d in v1["tags"].items():
        for ref in [tag] + d["aliases"]:
            url = f"{a.registry.rstrip('/')}/v2/grux/grux-local/manifests/{ref}"
            try:
                body, hdr = fetch(url, {"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
                got = "sha256:" + hashlib.sha256(body).hexdigest()
                served = {k.lower(): v for k, v in hdr.items()}.get("docker-content-digest")
                ok = got == d["manifest_digest"] == served
            except Exception as e:
                got, served, ok = f"error {e}", None, False
            failed += not ok
            print(f"{'PASS' if ok else 'FAIL'} registry {ref}: body {got[:19]}, header {str(served)[:19]}, "
                  f"v1.json {d['manifest_digest'][:19]} ({url})")
    if a.blobs:
        failed += not blob_burst(a.registry, v1, a.blobs)
    print("ALL PASS" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
