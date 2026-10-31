#!/usr/bin/env python3
"""Numbers check: every surface that states a Grux Local number must state the one in results/v1.json.

  python3 results/check.py                                  # the live surfaces and the live registry
  python3 results/check.py --card README.md --readme ../grux/README.md --site page.html --registry https://<host>

Each surface is a path or a URL; the blog is the dotcomjack.com announcement, checked on its prose. Four checks per surface, all from v1.json:
  rows   every table row and headline sentence that surface carries, matched whole, so a right number in
         the wrong row, or a row that went missing, fails;
  shape  every number on the surface shaped like one of ours ("N (P%)", "N of 464", "N of 466", "N out of
         70", "X.XX GB", "X.X s") must be one of ours, so a stray wrong number elsewhere on the page fails too.
  scope  numbers measured on another Mac than v1.json's "machine": a tag's before-training numbers when its
         before_training_machine differs (today the 9B's, on a MacBook Pro), and every number of a tag with
         its own "machine" (today the 4b-v2, all on that MacBook Pro: counts, shares, hidden counts, chat
         p95, loaded size, and the size Ollama reported after use, from loaded_after_use). Each must sit in a table whose header names that Mac and no other, or have
         that Mac's name within SCOPE_WINDOW characters with no other Mac named in between. A number no
         other Mac measured counts wherever it stands; a hidden count, and any number another Mac also
         measured, counts where that tag's label is the nearest one before it. A sentence that names the
         Mac mini without that Mac fails when it puts the counts on one Mac mini ("on one Mac mini", or
         "every"/"all" ... "on the 16 GB Mac mini"), speaks of a 9B-style tag's numbers before training, or
         names a 4b-v2-style tag. Machines are "Name, chip, N GB"; a machine in any other form, a tag that
         names no machine, or a second Mac that shares the Mac mini's name fails;
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
import collections
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


# A sentence that puts every count on the top-level Mac must name each other Mac a number came from. Found by
# the 2026-10-23 reviews of the 9B drafts: "Every task count here ... on one Mac mini." with the MacBook Pro cut,
# then "... on the 16 GB Mac mini." and "... all measured on a Mac mini." Likewise a sentence about a size's
# numbers before training that names the Mac mini ("Its before-training numbers are from a Mac mini ... 9B").
ONE_MACHINE = [r"\bon one {top}\b",
               r"\b(?:[Ee]very|[Aa]ll)\b.*\bon (?:one|a single|the same|that|the|a) (?:\d+ GB )?{top}\b"]
MACHINE_FORM = r"[^,()]+, [^,()]+, \d+ GB"   # "Mac mini, Apple M4, 16 GB"
SCOPE_WINDOW = 160


def machine_name(m):
    return m.split(", ")[0]   # "Mac mini, Apple M4, 16 GB" -> "Mac mini"


def labels_of(tag, d):
    """What names a tag in prose: its label ("4B v2") or size ("9B"), its tag ("grux-local:4b-v2"), and for a tag
    compared with v1's 4B on its own Mac, "v1 4B", which only that comparison uses."""
    out = [d.get("label", d["size"]), f"grux-local:{tag}"]
    return out + (["v1 4B"] if "v1_4b_same_machine" in d else [])


def runs_of(d, top):
    """(Mac, summary) for every scored run a tag carries."""
    mine = d.get("machine", top)
    out = [(mine, d["grux_local"])]
    if "before_training" in d:
        out.append((d.get("before_training_machine", top), d["before_training"]))
    if "grux_local_on_before_training_machine" in d:
        out.append((d.get("before_training_machine", top), d["grux_local_on_before_training_machine"]))
    if "v1_4b_same_machine" in d:
        out.append((mine, d["v1_4b_same_machine"]))
    for k in ("grux_local", "v1_4b_same_machine"):
        if k in d.get("holdout_v2", {}):
            out.append((mine, d["holdout_v2"][k]))
    return out


def baselines(v1):
    """Every number measured on another Mac than v1.json's "machine", grouped by tag, with the regexes that find it
    on a surface. Fails closed: a tag that names no machine, a machine not in "Name, chip, N GB" form, a second
    Mac that shares the top-level Mac's name, or another Mac with no scope check built is a problem, never a
    skipped tag."""
    top, t = v1["machine"], v1["tags"]
    if not re.fullmatch(MACHINE_FORM, top):
        return [], [f"machine {top!r} is not in the form 'Name, chip, N GB'"]
    problems = []
    for tag, d in t.items():
        if "machine" not in d and "before_training_machine" not in d:
            problems.append(f"{tag} names no machine (machine or before_training_machine)")
        for key in ("machine", "before_training_machine"):
            m = d.get(key)
            if m is None:
                continue
            if not re.fullmatch(MACHINE_FORM, m):
                problems.append(f"{tag} {key} {m!r} is not in the form 'Name, chip, N GB'")
            elif m != top and machine_name(m) == machine_name(top):
                problems.append(f"{tag} {key} {m!r} shares the name {machine_name(top)!r} with {top!r}")
    for e in v1.get("loaded_after_use", []):
        if not re.fullmatch(MACHINE_FORM, e["machine"]):
            problems.append(f"loaded_after_use {e['tag']} machine {e['machine']!r} is not in the form 'Name, chip, N GB'")
        elif e["machine"] != top and machine_name(e["machine"]) == machine_name(top):
            problems.append(f"loaded_after_use {e['tag']} machine {e['machine']!r} shares the name {machine_name(top)!r}")
    if problems:
        return [], problems
    # What each Mac measured, by kind; a number another Mac also measured is ambiguous. The chat percentile
    # (the "95%" in "95% came back within") stands on every surface, so it counts as the top-level Mac's share.
    seen = collections.defaultdict(lambda: {"n": set(), "pct": set(), "s": set(), "gb": set()})
    seen[machine_name(top)]["pct"].add(v1["chat_percentile"])
    for d in t.values():
        for m, r in runs_of(d, top):
            k = seen[machine_name(m)]
            k["n"] |= {r["pass"], r["hidden_instructions_ignored"]}
            k["pct"].add(pct(r["pass"], r["of"]))
            k["s"].add(f"{r['chat_p95_s']:.1f}")
        seen[machine_name(d.get("machine", top))]["gb"].add(f"{d['loaded_gb']:.2f}")
    for e in v1.get("loaded_after_use", []):
        seen[machine_name(e["machine"])]["gb"] |= {f"{e['fresh_gb']:.2f}", f"{e['after_gb']:.2f}"}

    def other(mac, kind, v):
        return any(v in k[kind] for name, k in seen.items() if name != mac)

    out = []
    for tag, d in t.items():
        if d.get("machine", top) != top:
            mac, kind = machine_name(d["machine"]), "all"
            runs = [r for _, r in runs_of(d, top)]
        elif d.get("before_training_machine", top) != top:
            mac, kind = machine_name(d["before_training_machine"]), "before"
            runs = [d["before_training"]]
        else:
            continue
        # (what, regex, only where this tag's label is the nearest one before it)
        nums = []
        for r in runs:
            p, h, sh = r["pass"], r["hidden_instructions_ignored"], pct(r["pass"], r["of"])
            # A bare number, but never a piece of a date, a path or a version ("2026-10-24", "v1.2").
            nums += [("count", rf"(?<![\d.,/-]){p}(?![\d/-]|[.,]\d|%)", other(mac, "n", p)),
                     ("share", rf"(?<![\d.,/-]){sh}%", other(mac, "pct", sh)),
                     ("hidden count", rf"(?<![\d.,/-]){h}(?![\d/-]|[.,]\d|%)", True)]
            if kind == "all":
                x = f"{r['chat_p95_s']:.1f}"
                nums.append(("chat p95", rf"(?<![\d.]){re.escape(x)} ?s(?:econds)?\b", other(mac, "s", x)))
        if kind == "all":
            g = f"{d['loaded_gb']:.2f}"
            nums.append(("loaded size", rf"(?<![\d.]){re.escape(g)} GB\b", other(mac, "gb", g)))
        nums = list(dict.fromkeys(nums))
        out.append({"tag": tag, "labels": labels_of(tag, d), "mac": mac, "kind": kind, "nums": nums})
    # A size read after use on another Mac is that tag's number too, scoped like the rest of it.
    for u in v1.get("loaded_after_use", []):
        if u["machine"] == top:
            continue
        mac, g = machine_name(u["machine"]), f"{u['after_gb']:.2f}"
        entry = next((e for e in out if e["tag"] == u["tag"] and e["mac"] == mac), None)
        if entry is None:
            problems.append(f"loaded_after_use {u['tag']} was measured on {u['machine']!r}, not {top!r}, and no scope "
                            "check was built for it")
            continue
        entry["nums"].append(("loaded size after use", rf"(?<![\d.]){re.escape(g)} GB\b", other(mac, "gb", g)))
    built = {e["tag"] for e in out if e["nums"]}
    for tag, d in t.items():
        for key in ("machine", "before_training_machine"):
            if d.get(key, top) != top and tag not in built:
                problems.append(f"{tag} was measured on {d[key]!r} ({key}), not {top!r}, and no scope check was "
                                "built for it")
    return out, problems


def flat(s, is_html):
    """The text_of flattening, for a piece of a surface."""
    if is_html:
        s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    else:
        s = re.sub(r"^\s*>\s?", "", s, flags=re.M).replace("**", "").replace("`", "").replace("|", " ")
    return re.sub(r"\s+", " ", s).strip()


def table_spans(raw, text):
    """(start, end, header) of every table on the surface, in text_of's coordinates."""
    s = raw.decode("utf-8", "replace")
    is_html = "<html" in s[:2000].lower()
    if is_html:
        found = [(m.group(0), (re.search(r"<thead\b.*?</thead>|<tr\b.*?</tr>", m.group(0), re.S | re.I) or m).group(0))
                 for m in re.finditer(r"<table\b.*?</table>", s, re.S | re.I)]
    else:
        found = [(m.group(0), m.group(0).splitlines()[0]) for m in re.finditer(r"(?m)^\|.*\n(?:^\|.*(?:\n|$))+", s)]
    out = []
    for chunk, head in found:
        body = flat(chunk, is_html)
        i = text.find(body)
        if body and i >= 0:
            out.append((i, i + len(body), flat(head, is_html)))
    return out


def unscoped(text, sentences, tables, scoped, v1):
    """Every number in `scoped` must sit in a table whose header names its Mac and no other, or have its Mac named
    within SCOPE_WINDOW characters with no other Mac named in between; returns (numbers checked, failures)."""
    top = machine_name(v1["machine"])
    macs = {top} | {e["mac"] for e in scoped}
    names = sorted(((l, tag) for tag, d in v1["tags"].items() for l in labels_of(tag, d)), key=lambda x: -len(x[0]))
    owner = dict(names)
    label_rx = "|".join(rf"\b{re.escape(l)}(?![\w-])" if l.startswith("grux-local:") else rf"\b{re.escape(l)}\b"
                        for l, _ in names)
    n, bad, said = 0, [], set()
    for e in scoped:
        name, others = e["mac"], macs - {e["mac"]}
        for what, rx, under_label in e["nums"]:
            for m in re.finditer(rx, text):
                i, j = m.start(), m.end()
                if under_label:
                    before = re.findall(label_rx, text[max(0, i - 300):i])
                    if not before or owner[before[-1]] != e["tag"]:
                        continue
                n += 1
                if any(a <= i < z and name in h and not any(o in h for o in others) for a, z, h in tables):
                    continue
                lo = max(0, i - SCOPE_WINDOW)
                named = False
                for k in re.finditer(re.escape(name), text[lo:j + SCOPE_WINDOW]):
                    a, z = lo + k.start(), lo + k.end()
                    between = text[z:i] if z <= i else text[j:a]
                    named |= not any(o in between for o in others)
                if not named:
                    bad.append(f"{e['tag']} {what} {m.group(0)!r} without the {name} next to it: "
                               f"...{text[max(0, i - 70):j + 30]}...")
        own = [l for l in e["labels"] if l != "v1 4B"]
        for sentence in sentences:
            if name in sentence or top not in sentence:
                continue
            if any(re.search(p.format(top=re.escape(top)), sentence) for p in ONE_MACHINE):
                if (sentence, name) not in said:
                    said.add((sentence, name))
                    bad.append(f"puts every count on one {top} without the {name}: {sentence[:140]!r}")
            elif e["kind"] == "before" and re.search(rf"\b{re.escape(own[0])}\b", sentence) \
                    and re.search(r"\bbefore[ -]training\b", sentence, re.I):
                bad.append(f"{own[0]} before training beside the {top} without the {name}: {sentence[:140]!r}")
            elif e["kind"] == "all" and any(re.search(rf"\b{re.escape(l)}(?![\w-])", sentence) for l in own):
                bad.append(f"{own[0]} beside the {top} without the {name}: {sentence[:140]!r}")
    return n, bad


def pct(n, of):
    return round(100 * n / of)


def behind(a, b):
    """How the surface says a is behind b: "one task behind", "2 tasks behind"."""
    gap = b - a
    return "one task behind" if gap == 1 else f"{gap} tasks behind"


def own_machine_rows(tag, d):
    """The rows a tag measured on its own Mac carries on the card and the site: its tag and loaded size, then
    each held-out set against v1's 4B on the same Mac (that column first, as "before" is elsewhere), the
    realistic chat p95 of both, and, when it trails v1 on the original set, that it does."""
    L, o, c = d["label"], d["grux_local"], d["v1_4b_same_machine"]
    o2, c2 = d["holdout_v2"]["grux_local"], d["holdout_v2"]["v1_4b_same_machine"]
    of, of2, hc = o["of"], o2["of"], o["hidden_instruction_cases"]
    rows = [
        ("loaded", f"{L} loaded {d['loaded_gb']:.2f} GB on {d['made_for']}",
         rf"grux-local:{re.escape(tag)}(?![\w-])[^0-9]{{0,40}}?{re.escape(d['made_for'])} {d['loaded_gb']:.2f} GB"),
        ("all", f"{L} held-out v1: {c['pass']} to {o['pass']} of {of}",
         rf"Held-out v1, all (?:{of} )?tasks {c['pass']} \({pct(c['pass'], of)}%\) {o['pass']} \({pct(o['pass'], of)}%\)"),
        ("hidden", f"{L} held-out v1 hidden: {c['hidden_instructions_ignored']} to {o['hidden_instructions_ignored']} of {hc}",
         rf"Held-out v1, hidden instructions ignored {c['hidden_instructions_ignored']} of {hc} "
         rf"{o['hidden_instructions_ignored']} of {hc}"),
        ("all", f"{L} held-out v2: {c2['pass']} to {o2['pass']} of {of2}",
         rf"Held-out v2, all (?:{of2} )?tasks {c2['pass']} \({pct(c2['pass'], of2)}%\) {o2['pass']} \({pct(o2['pass'], of2)}%\)"),
        ("hidden", f"{L} held-out v2 hidden: {c2['hidden_instructions_ignored']} to {o2['hidden_instructions_ignored']} of {hc}",
         rf"Held-out v2, hidden instructions[^0-9]{{0,60}}? {c2['hidden_instructions_ignored']} of {hc} "
         rf"{o2['hidden_instructions_ignored']} of {hc}"),
        ("chat", f"{L} realistic chat p95 {o2['chat_p95_s']:.1f} s",
         rf"{o2['chat_p95_s']:.1f} s(?:econds)?(?: on the {L}| \({L}\))"),
        ("chat", f"v1 4B realistic chat p95 {c2['chat_p95_s']:.1f} s",
         rf"{c2['chat_p95_s']:.1f} s(?:econds)?(?: on (?:the )?v1 4B| \(v1 4B\))"),
    ]
    if o["pass"] < c["pass"]:
        rows.append(("behind", f"{L} {behind(o['pass'], c['pass'])} v1 on held-out v1",
                     rf"{behind(o['pass'], c['pass'])}\b[^.]{{0,80}}?\b{o['pass']} against {c['pass']}\b"))
    return rows


def own_machine_headline(d):
    """The Grux README's one line for a tag measured on its own Mac."""
    o, c = d["grux_local"], d["v1_4b_same_machine"]
    o2, c2 = d["holdout_v2"]["grux_local"], d["holdout_v2"]["v1_4b_same_machine"]
    out = [("headline", f"{d['label']} held-out v2 line",
            rf"\bit got {o2['pass']}, against {c2['pass']} for v1's 4B\b")]
    if o["pass"] < c["pass"]:
        out.append(("headline", f"{d['label']} held-out v1 line",
                    rf"{behind(o['pass'], c['pass'])}\b[^.]{{0,80}}?\b{o['pass']} against {c['pass']}\b"))
    return out


def expected(v1):
    t = v1["tags"]
    rows, own, allowed = [], [], {"pairs": set(), "of_tasks": collections.defaultdict(set), "of_hidden": set(),
                                  "gb": set(), "s": set()}
    for tag, d in t.items():
        if "v1_4b_same_machine" in d:   # a tag measured on its own Mac (its machine is checked in baselines)
            own += own_machine_rows(tag, d)
            for r in [r for _, r in runs_of(d, v1["machine"])]:
                allowed["pairs"].add((r["pass"], pct(r["pass"], r["of"])))
                allowed["of_tasks"][r["of"]].add(r["pass"])
                allowed["of_hidden"].add(r["hidden_instructions_ignored"])
                allowed["s"].add(f"{r['chat_p95_s']:.1f}")
            allowed["gb"].add(f"{d['loaded_gb']:.2f}")
            continue
        if "before_training" not in d:
            continue
        T, a, b = d["size"], d["grux_local"], d["before_training"]
        of, hc = a["of"], a["hidden_instruction_cases"]
        ap, bp = pct(a["pass"], of), pct(b["pass"], of)
        rows += [
            ("all", f"{T} all tasks: {b['pass']} ({bp}%) to {a['pass']} ({ap}%)",
             rf"{T}, all (?:{of} )?tasks {b['pass']} \({bp}%\) {a['pass']} \({ap}%\)"),
            ("hidden", f"{T} hidden instructions: {b['hidden_instructions_ignored']} to {a['hidden_instructions_ignored']} of {hc}",
             rf"{T}, hidden instructions ignored {b['hidden_instructions_ignored']} of {hc} {a['hidden_instructions_ignored']} of {hc}"),
            ("loaded", f"{T} loaded {d['loaded_gb']:.2f} GB on {d['made_for']}",
             rf"grux-local:{re.escape(tag)}(?![\w-])[^0-9]{{0,40}}?{re.escape(d['made_for'])} {d['loaded_gb']:.2f} GB"),
            ("chat", f"{T} chat p95 {a['chat_p95_s']:.1f} s",
             rf"{a['chat_p95_s']:.1f} s(?:econds)?(?: on the {T}| \({T}\))"),
        ]
        allowed["pairs"] |= {(a["pass"], ap), (b["pass"], bp)}
        allowed["of_tasks"][of] |= {a["pass"], b["pass"]}
        allowed["of_hidden"] |= {a["hidden_instructions_ignored"], b["hidden_instructions_ignored"]}
        allowed["gb"].add(f"{d['loaded_gb']:.2f}")
        allowed["s"].add(f"{a['chat_p95_s']:.1f}")
    for u in v1.get("loaded_after_use", []):
        allowed["gb"] |= {f"{u['fresh_gb']:.2f}", f"{u['after_gb']:.2f}"}
    big, small, nine = t["4b"], t["2b"], t["9b"]
    of = big["grux_local"]["of"]
    v2 = t.get("4b-v2")
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
        ] + (own_machine_headline(v2) if v2 else []),
        "site": [
            ("headline", "4B lead", rf"the 4B got {big['grux_local']['pass']} right, against "
                                    rf"{big['before_training']['pass']} for the same model before training"),
        ] + own,
        # The model card carries the whole 4b-v2 section; the Grux README one line (above).
        "card": own,
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
    for denominator, ok in sorted(allowed["of_tasks"].items()):
        for m in re.finditer(rf"\b(\d+) of {denominator}\b", text):
            n += 1
            if int(m.group(1)) not in ok:
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
    layers, missing = [], []
    for tag, d in v1["tags"].items():
        try:   # a tag the registry does not serve yet is a FAIL line, not a traceback that hides the rest
            body, _ = fetch(f"{registry.rstrip('/')}/v2/grux/grux-local/manifests/{tag}",
                            {"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
        except Exception as e:
            missing.append(f"{tag} ({e})")
            continue
        layers += [(l["digest"], l["size"]) for l in json.loads(body)["layers"] if l["size"] > 4096]
    layers = sorted(set(layers), key=lambda x: -x[1])
    if not layers:   # every manifest failed: still a FAIL line and the summary, never a ZeroDivisionError
        print(f"FAIL blobs: no layers to fetch, no manifest for {', '.join(missing)}")
        return False
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
    ok = len(bad) <= n * 0.05 and not missing
    print(f"{'PASS' if ok else 'FAIL'} blobs: {n - len(bad)} of {n} ranged GETs answered 206 across "
          f"{len(layers)} layers" + (f", failures {sorted(bad)}" if bad else "")
          + (f", no manifest for {', '.join(missing)}" if missing else ""))
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
        n_scope, loose = unscoped(text, sentences_of(ours), table_spans(ours, text), scoped, v1)
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
            print(f"{'PASS' if ok else 'FAIL'} registry {ref}: body {got if got.startswith('error') else got[:19]}, header {str(served)[:19]}, "
                  f"v1.json {d['manifest_digest'][:19]} ({url})")
    if a.blobs:
        failed += not blob_burst(a.registry, v1, a.blobs)
    print("ALL PASS" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
