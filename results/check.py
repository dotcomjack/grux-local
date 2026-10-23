#!/usr/bin/env python3
"""Numbers check: every surface that states a Grux Local number must state the one in results/v1.json.

  python3 results/check.py                                  # the live surfaces and the live registry
  python3 results/check.py --card README.md --readme ../grux/README.md --site page.html --registry https://<host>

Each surface is a path or a URL; the blog is the dotcomjack.com announcement, checked on its prose. Four checks per surface, all from v1.json:
  rows   every table row and headline sentence that surface carries, matched whole, so a right number in
         the wrong row, or a row that went missing, fails;
  shape  every number on the surface shaped like one of ours ("N (P%)", "N of 464", "N out of 70",
         "X.XX GB", "X.X s") must be one of ours, so a stray wrong number elsewhere on the page fails too.
  scope  a tag whose before_training_machine is not v1.json's "machine" (today the 9B, on a MacBook Pro):
         every one of its before-training numbers on the surface must have that Mac's name within
         SCOPE_WINDOW characters with no other Mac named in between; and a sentence that names the Mac mini
         and either puts the counts "on one Mac mini" or speaks of that tag's numbers before training must
         name that Mac too, so a baseline never reads as the Mac mini's. A tag with no
         before_training_machine, or a second Mac that shares the first one's name, fails;
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


def sentences_of(raw):
    """The surface's sentences, each table cell, list item, heading and paragraph its own block first, so a
    sentence never runs on across a table into the next paragraph."""
    s = raw.decode("utf-8", "replace")
    if "<html" in s[:2000].lower():
        m = re.search(r"<main\b.*?</main>", s, re.S | re.I)
        s = m.group(0) if m else s
        s = re.sub(r"<(script|style)\b.*?</\1>", " ", s, flags=re.S | re.I)
        s = re.sub(r"</(?:p|td|th|li|dd|dt|h\d|caption)>|<br\s*/?>", "\n\n", s, flags=re.I)
        s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    else:
        s = re.sub(r"^\s*>\s?", "", s, flags=re.M)
        s = s.replace("**", "").replace("`", "").replace("|", "\n\n")
    out = []
    for block in re.split(r"\n\s*\n", s):
        out += [x for x in re.split(r"(?<=[.!?:])\s+", re.sub(r"\s+", " ", block).strip()) if x]
    return out


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


# A sentence that puts every count on the top-level Mac must name each other Mac a baseline came from. Found by
# the 2026-10-23 review of the 9B drafts: "Every task count here ... on one Mac mini." with the MacBook Pro cut.
# Likewise a sentence about a size's numbers before training that names the Mac mini ("Its before-training
# numbers are from a Mac mini ... the stock 9B").
ONE_MACHINE = [r"\bon one {top}\b", r"\bevery (?:number|task count)\b[^.]*\bon (?:one|a single|the same) {top}\b"]
SCOPE_WINDOW = 160


def machine_name(m):
    return m.split(", ")[0]   # "Mac mini, Apple M4, 16 GB" -> "Mac mini"


def baselines(v1):
    """Each tag whose before-training numbers come from another Mac than v1.json's "machine", with the regexes
    that find those numbers on a surface. Fails closed: a tag with no before_training_machine, or whose Mac
    shares the top-level Mac's name, is a problem, never a skipped tag."""
    top, t = v1["machine"], v1["tags"]
    here = set()   # numbers measured on the top-level Mac; a baseline number equal to one of them is ambiguous
    for d in t.values():
        here |= {d["grux_local"]["pass"], d["grux_local"]["hidden_instructions_ignored"]}
        if d.get("before_training_machine") == top:
            here |= {d["before_training"]["pass"], d["before_training"]["hidden_instructions_ignored"]}
    out, problems = [], []
    for tag, d in t.items():
        m = d.get("before_training_machine")
        if not m:
            problems.append(f"{tag} has no before_training_machine")
            continue
        if m == top:
            continue
        if machine_name(m) == machine_name(top):
            problems.append(f"{tag} before_training_machine {m!r} shares the name {machine_name(top)!r} with {top!r}")
            continue
        b, of, hc = d["before_training"], d["grux_local"]["of"], d["grux_local"]["hidden_instruction_cases"]
        p, h = b["pass"], b["hidden_instructions_ignored"]
        # (label, regex, only where the nearest size label before it is this tag's): a number that is this
        # baseline's alone counts wherever it stands; one the Mac mini also measured counts only under the size.
        nums = [("before-training count", rf"(?<![\d.,]){p}(?!\d|[.,]\d)", False) if p not in here else
                ("before-training count", rf"\b{p} (?:\(\d+%\)|of {of}\b)", True),
                ("before-training hidden", rf"\b{h} (?:times )?(?:out )?of {hc}\b", h in here)]
        out.append((d["size"], machine_name(m), nums))
    built = {size for size, _, nums in out if nums}
    for tag, d in t.items():
        if d.get("before_training_machine", top) != top and d["size"] not in built:
            problems.append(f"{tag} was measured before training on {d['before_training_machine']!r}, not {top!r}, "
                            "and no scope check was built for it")
    return out, problems


def unscoped(text, sentences, scoped, v1):
    """Every baseline number in `scoped` must have its Mac named within SCOPE_WINDOW characters, with no other
    Mac named in between; returns (numbers checked, failures)."""
    top = machine_name(v1["machine"])
    macs = {top} | {name for _, name, _ in scoped}
    sizes = "|".join(re.escape(d["size"]) for d in v1["tags"].values())
    n, bad = 0, []
    for size, name, nums in scoped:
        others = macs - {name}
        for label, rx, under_size in nums:
            for m in re.finditer(rx, text):
                i, j = m.start(), m.end()
                if under_size:
                    before = re.findall(rf"\b({sizes})\b", text[max(0, i - 300):i])
                    if not before or before[-1] != size:
                        continue
                n += 1
                lo = max(0, i - SCOPE_WINDOW)
                named = False
                for k in re.finditer(re.escape(name), text[lo:j + SCOPE_WINDOW]):
                    a, z = lo + k.start(), lo + k.end()
                    between = text[z:i] if z <= i else text[j:a]
                    named |= not any(o in between for o in others)
                if not named:
                    bad.append(f"{size} {label} {m.group(0)!r} without the {name} next to it: "
                               f"...{text[max(0, i - 70):j + 30]}...")
        for sentence in sentences:
            if name in sentence or top not in sentence:
                continue
            if any(re.search(p.format(top=re.escape(top)), sentence) for p in ONE_MACHINE):
                bad.append(f"puts every count on one {top} without the {name}: {sentence[:140]!r}")
            elif re.search(rf"\b{re.escape(size)}\b", sentence) and re.search(r"\bbefore[ -]training\b", sentence, re.I):
                bad.append(f"{size} before training beside the {top} without the {name}: {sentence[:140]!r}")
    return n, bad


def pct(n, of):
    return round(100 * n / of)


def expected(v1):
    t = v1["tags"]
    rows, allowed = [], {"pairs": set(), "of_tasks": set(), "of_hidden": set(), "gb": set(), "s": set()}
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
        ],
        "site": [
            ("headline", "4B lead", rf"the 4B got {big['grux_local']['pass']} right, against "
                                    rf"{big['before_training']['pass']} for the same model before training"),
        ],
        # The dotcomjack.com announcement (2026-10-23) carries prose, not the tables.
        "blog": [
            ("headline", "blog task count", rf"across {of} held-out Grux tasks"),
            ("headline", "blog 4B", rf"the 4B version got {big['grux_local']['pass']} right, against "
                                    rf"{big['before_training']['pass']} for the same model before training"),
            ("headline", "blog 2B", rf"2B version, made for 8 GB Macs, went from {small['before_training']['pass']} "
                                    rf"to {small['grux_local']['pass']}\b"),
            ("headline", "blog 9B", rf"9B version, made for {re.escape(nine['made_for'])}, went from "
                                    rf"{nine['before_training']['pass']} to {nine['grux_local']['pass']}\b"),
        ],
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
    scoped, problems = baselines(v1)
    failed = len(problems)
    for p in problems:
        print(f"FAIL v1.json: {p}")
    for name in ("card", "readme", "site", "blog"):
        src = getattr(a, name)
        try:
            raw, _ = fetch(src)
        except Exception as e:
            print(f"FAIL {name}: could not read {src}: {e}")
            failed += 1
            continue
        md = raw.decode("utf-8", "replace")
        ours = readme_scope(md).encode() if name == "readme" else raw
        text = text_of(ours, src)
        want = ([] if name == "blog" else rows) + headline.get(name, [])
        miss = [label for _, label, rx in want if not re.search(rx, text)]
        n, bad = shape(text, allowed, of, hc)
        claims = absolutes(text, v1)
        n_scope, loose = unscoped(text, sentences_of(ours), scoped, v1)
        # Prose (the blog) has no table shapes, so there every row matching is the floor instead.
        ok = not miss and not bad and not claims and not loose and (n > 0 or (name == "blog" and bool(want)))
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {len(want) - len(miss)} of {len(want)} rows match, "
              f"{n} numbers checked by shape, {len(bad)} wrong, {len(claims)} absolute claims, "
              f"{n_scope} other-Mac numbers checked, {len(loose)} unscoped ({src})")
        for label in miss:
            print(f"     missing or different: {label}")
        for b in bad:
            print(f"     not a v1.json number: {b!r}")
        for c in claims:
            print(f"     absolute claim: {c}")
        for c in loose:
            print(f"     unscoped: {c}")
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
