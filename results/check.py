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
The registry check ties the numbers to the files: each tag's manifest must hash to v1.json's digest. The pullpath
check walks Ollama 0.40.8's GGUF pull (HEAD same-host, GET with a Location that answers 206) for each GGUF tag.
--blobs N fires N ranged GETs at once across every blob of every tag, the way `ollama pull` does, and
fails if more than 5% do not answer 206: on 2026-10-23 the preview passed every check while production
dropped about 1 blob request in 10 at the upstream, which only a burst of real ranged GETs showed.
A tag that replaced an earlier build (today the 2B, rebuilt 2026-11-24) is checked as it is now on every surface but
the blog, which is dated: the blog is checked against v1.json as it stood when it was written, with each tag in
"replaced" back in place of the tag that replaced it, so the old 2B's numbers pass there and fail everywhere else.
A tag whose "input" is "text only" must be said to be text only on the card, the Grux README and the site, with the
image attachment that fails in Grux named; on the card, its comparison with the build it replaced must carry every
row, and every place it is worse. A tag republished with only a parameter changed (the 2B on 2026-11-27, num_batch
256: its "num_batch_change") must carry, on the card, its comparison with the build before it, every local turn that
moved, every place it is worse and the waiver that let it ship; on the Grux README, one line of what it did.
Exit 0 only when every surface passes; a surface with nothing matched is a FAIL, never a pass.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import html
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.parse
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
    # The callout is the blockquote whose bold opening names Grux Local. Matched by that, never by its exact first
    # words: 3.2 renamed "New: Grux Local" to "Grux Local, the model trained for Grux, ...", and the old pattern
    # cut the headline out of the check, red for 10 days (2026-11-01 to 11-11) with every number right.
    callout = re.search(r"^> \*\*[^\n]*?Grux Local.*?(?=^[^>]|\Z)", md, re.S | re.M)
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
    if "previous_2b_same_machine" in d:
        out.append((mine, d["previous_2b_same_machine"]))
    for k in ("grux_local", "v1_4b_same_machine", "previous_2b_same_machine"):
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
            # A bare number, but never a piece of a date, a path or a version ("2026-10-24", "v1.2"), nor an HTTP
            # status ("HTTP 400", the text-only 2B's answer to an image).
            nums += [("count", rf"(?<![\d.,/-])(?<!HTTP ){p}(?![\d/-]|[.,]\d|%)", other(mac, "n", p)),
                     ("share", rf"(?<![\d.,/-]){sh}%", other(mac, "pct", sh)),
                     ("hidden count", rf"(?<![\d.,/-])(?<!HTTP ){h}(?![\d/-]|[.,]\d|%)", True)]
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


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
# How the card says each reason the scorer gives for a failed chat-honesty turn ("acted: ['create_event']").
HONESTY_REASONS = {"empty reply": ("empty reply", "empty replies"),
                   "acted": ("event created that nobody asked for", "events created that nobody asked for"),
                   "called a tool this turn did not offer": ("call to a tool the turn did not offer",
                                                             "calls to a tool the turn did not offer"),
                   "claims an action no offered tool does": ("offer of an action no offered tool does",
                                                             "offers of actions no offered tool does")}
# The sets a first word is timed on, as the card names them.
FIRST_WORD = [("holdout_v1", "held-out v1"), ("holdout_v2", "held-out v2"),
              ("local_num_ctx_8192", "the local turns at num_ctx 8,192")]


def span(xs):
    """"163 to 166", or "27" when every run gave the same number."""
    lo, hi = min(xs), max(xs)
    return f"{lo}" if lo == hi else f"{lo} to {hi}"


def runs_said(rs):
    """"163 to 166 (four runs)", "196 (one run)": the totals of a list of local-turn runs and how many there were."""
    return rf"{span([r['pass'] for r in rs])} \({WORDS[len(rs)]} runs?\)"


def text_only_rows(tag, d):
    """A text-only tag must be said to be text only, and the Grux chat that fails with it named."""
    if d.get("input") != "text only":
        return []
    T = re.escape(d["size"])
    return [("text only", f"{d['size']} text only",
             rf"grux-local:{re.escape(tag)}(?![\w-])[^0-9.]{{0,40}}?\btext only\b|\b{T}\b[^.]{{0,60}}?\btext only\b"),
            ("text only", f"{d['size']} image attachment fails in Grux",
             rf"\bimage attached fails\b[^.]{{0,60}}?\blocal {T}\b")]


def rebuilt_rows(tag, d):
    """The card's comparison of a rebuilt tag with the first build of that tag, on the same Mac: every row, and every
    place the new build is worse, from v1.json. Each side's local-turn runs are listed apart (the two builds were not
    run in pairs), and a slower chat or a later first word is said for exactly the sets where it is."""
    T, a, b = d["size"], d["grux_local"], d["previous_2b_same_machine"]
    a2, b2 = d["holdout_v2"]["grux_local"], d["holdout_v2"]["previous_2b_same_machine"]
    of, of2, hc = a["of"], a2["of"], a["hidden_instruction_cases"]
    la, lb = d["local_turns"]["grux_local"], d["local_turns"]["previous_2b_same_machine"]
    ha = [r["families"]["chat_honesty"]["pass"] for r in la]
    hb = [r["families"]["chat_honesty"]["pass"] for r in lb]
    hof = la[0]["families"]["chat_honesty"]["of"]
    fa, fb = d["file_naming"]["grux_local"], d["file_naming"]["previous_2b_same_machine"]
    ft = d["first_token_median_s"]
    fails = d["local_turns"]["chat_honesty_failures_last_run"]
    rows = [
        ("rebuilt", f"{T} rebuilt, held-out v1: {b['pass']} to {a['pass']} of {of}",
         rf"Held-out v1, all (?:{of} )?tasks {b['pass']} \({pct(b['pass'], of)}%\) {a['pass']} \({pct(a['pass'], of)}%\)"),
        ("rebuilt", f"{T} rebuilt, held-out v1 hidden: {b['hidden_instructions_ignored']} to {a['hidden_instructions_ignored']}",
         rf"Held-out v1, hidden instructions ignored {b['hidden_instructions_ignored']} of {hc} "
         rf"{a['hidden_instructions_ignored']} of {hc}"),
        ("rebuilt", f"{T} rebuilt, held-out v2: {b2['pass']} to {a2['pass']} of {of2}",
         rf"Held-out v2, all (?:{of2} )?tasks {b2['pass']} \({pct(b2['pass'], of2)}%\) {a2['pass']} \({pct(a2['pass'], of2)}%\)"),
        ("rebuilt", f"{T} rebuilt, held-out v2 hidden: {b2['hidden_instructions_ignored']} to {a2['hidden_instructions_ignored']}",
         rf"Held-out v2, hidden instructions[^0-9]{{0,60}}? {b2['hidden_instructions_ignored']} of {hc} "
         rf"{a2['hidden_instructions_ignored']} of {hc}"),
        ("rebuilt", f"{T} rebuilt, local turns: {span([r['pass'] for r in lb])} to {span([r['pass'] for r in la])}",
         rf"Local turns.{{0,60}}? {la[0]['of']} tasks {runs_said(lb)} {runs_said(la)}"),
        ("rebuilt", f"{T} rebuilt, honesty on local turns: {span(hb)} to {span(ha)} of {hof}",
         rf"what it cannot do {span(hb)} of {hof} {span(ha)} of {hof}"),
        ("rebuilt", f"{T} rebuilt, file naming: {fb['pass']} to {fa['pass']} of {fa['of']}",
         rf"File naming[^0-9]{{0,30}}?8,192, {fa['of']} items {fb['pass']} of {fb['of']} {fa['pass']} of {fa['of']}"),
        ("rebuilt", f"{T} rebuilt, chat p95 on held-out v1: {b['chat_p95_s']:.1f} to {a['chat_p95_s']:.1f} s",
         rf"Chat replies on held-out v1, 95% within {b['chat_p95_s']:.1f} s {a['chat_p95_s']:.1f} s"),
        ("rebuilt", f"{T} rebuilt, chat p95 on held-out v2: {b2['chat_p95_s']:.1f} to {a2['chat_p95_s']:.1f} s",
         rf"Chat replies on held-out v2, 95% within {b2['chat_p95_s']:.1f} s {a2['chat_p95_s']:.1f} s"),
    ] + [("rebuilt", f"{T} rebuilt, first word on {name}",
          rf"First word on {re.escape(name)}, median {ft[k]['previous_2b_same_machine']:.2f} s {ft[k]['grux_local']:.2f} s")
         for k, name in FIRST_WORD] + [
        ("rebuilt", f"{T} rebuilt, the {sum(fails.values())} honesty failures by reason",
         rf"failed {sum(fails.values())} of the {hof} that ask for something no tool can do, in every run \("
         # In v1.json's order; a reason with no wording here is a KeyError, never a reason left off.
         + ", ".join(f"{n} {HONESTY_REASONS[k][n != 1]}" for k, n in fails.items()) + r"\)"),
        ("rebuilt", f"{T} rebuilt, runner footprint after use",
         rf"\b{d['runner_footprint_after_use_mb']:,} MB\b"),
        ("rebuilt", f"{T} rebuilt, its manifest", re.escape(d["manifest_digest"])),
        ("rebuilt", f"{T} rebuilt, its training gate",
         rf"\b{d['training_gate']['rows']:,} training rows\b.{{0,200}}?\b"
         + ("found none touched" if d["training_gate"]["held_out_touched"] == 0 else "(?!)")),
    ]
    if a["hidden_instructions_ignored"] < b["hidden_instructions_ignored"]:
        rows.append(("worse", f"{T} rebuilt ignored fewer hidden instructions on held-out v1",
                     rf"\b{a['hidden_instructions_ignored']} against {b['hidden_instructions_ignored']}\b"))
    slow = [n for n, x, y in (("held-out v1", a, b), ("held-out v2", a2, b2)) if x["chat_p95_s"] > y["chat_p95_s"]]
    if slow:
        rows.append(("worse", f"{T} rebuilt is slower on {' and '.join(slow)} chat",
                     rf"chat replies were slower on {' and on '.join(slow)}\b"))
    late = [n for k, n in FIRST_WORD if ft[k]["grux_local"] > ft[k]["previous_2b_same_machine"]]
    if late:
        rows.append(("worse", f"{T} rebuilt says its first word later on {' and '.join(late)}",
                     rf"first word came later on {re.escape(' and on '.join(late))}\b"))
    if fa["pass"] < fb["pass"]:
        rows.append(("worse", f"{T} rebuilt names fewer files", rf"\b{fa['pass']} against {fb['pass']} of {fa['of']}\b"))
    again = d["holdout_v1_second_run"]
    if again["flips"] == 0 and again["pass"] == a["pass"]:
        rows.append(("same", f"{T} rebuilt: its two held-out v1 runs agree",
                     r"[Tt]he two held-out v1 runs gave the same pass or fail on every task"))
    else:   # a surface may not say the two runs agree then
        rows.append(("same", f"{T} rebuilt: its two held-out v1 runs differ on {again['flips']} tasks", r"(?!)"))
    return rows


def num_batch_rows(tag, d):
    """The card's comparison of the tag with the build it replaced when only a parameter changed (the 2B's num_batch,
    2026-11-27): every row, every case that moved on the local turns, and every place the new build is worse, from
    v1.json's num_batch_change. A surface that says the two builds agree on a set where they do not cannot pass."""
    c, T = d["num_batch_change"], d["size"]
    o1, n1 = c["holdout_v1"]["replaced"], c["holdout_v1"]["grux_local"]
    o2, n2 = c["holdout_v2"]["replaced"], c["holdout_v2"]["grux_local"]
    lo, ln = c["local_turns"]["replaced"], c["local_turns"]["grux_local"]
    o8, n8 = [r for r in lo if r["num_ctx"] == 8192], [r for r in ln if r["num_ctx"] == 8192]
    o16, n16 = [r for r in lo if r["num_ctx"] == 16384], [r for r in ln if r["num_ctx"] == 16384]
    assert len(o8) == len(n8) == 1, "one run of each at num_ctx 8,192"
    o8, n8 = o8[0], n8[0]
    hc8 = n8["hidden_instruction_cases"]
    ft, fo, fn = c["first_token_median_s"], c["file_naming"]["replaced"], c["file_naming"]["grux_local"]
    gb_o, gb_n = c["loaded_gb"]["replaced"], c["loaded_gb"]["grux_local"]
    moved, b128 = c["local_turns"]["num_ctx_8192"], c["local_turns"]["num_batch_128_num_ctx_8192"]
    rows = [
        ("num_batch", f"{T} num_batch: the manifest it replaced", re.escape(c["replaced_manifest"])),
        ("num_batch", f"{T} num_batch: the parameter", rf"\bnum_batch {d['params']['num_batch']}\b"),
        ("num_batch", f"{T} num_batch: held-out v1 {o1['pass']} and {n1['pass']}",
         rf"Held-out v1, all {o1['of']} tasks {o1['pass']} \({pct(o1['pass'], o1['of'])}%\) "
         rf"{n1['pass']} \({pct(n1['pass'], n1['of'])}%\)"),
        ("num_batch", f"{T} num_batch: held-out v2 {o2['pass']} and {n2['pass']}",
         rf"Held-out v2, all {o2['of']} tasks {o2['pass']} \({pct(o2['pass'], o2['of'])}%\) "
         rf"{n2['pass']} \({pct(n2['pass'], n2['of'])}%\)"),
        ("num_batch", f"{T} num_batch: local turns at 8,192, {o8['pass']} and {n8['pass']}",
         rf"Local turns at num_ctx 8,192, {n8['of']} tasks {o8['pass']} {n8['pass']}"),
        ("num_batch", f"{T} num_batch: their hidden instructions, {o8['hidden_instructions_ignored']} and "
                      f"{n8['hidden_instructions_ignored']} of {hc8}",
         rf"Of those, hidden instructions[^0-9]{{0,60}}? {o8['hidden_instructions_ignored']} of {hc8} "
         rf"{n8['hidden_instructions_ignored']} of {hc8}"),
        ("num_batch", f"{T} num_batch: local turns at 16,384",
         rf"Local turns at num_ctx 16,384, {n8['of']} tasks {runs_said(o16)} {runs_said(n16)}"),
        ("num_batch", f"{T} num_batch: loaded {gb_o:.2f} and {gb_n:.2f} GB",
         rf"Freshly loaded at num_ctx 16,384 {gb_o:.2f} GB {gb_n:.2f} GB"),
        ("num_batch", f"{T} num_batch: file naming {fo['pass']} and {fn['pass']} of {fn['of']}",
         rf"File naming at num_ctx 8,192, {fn['of']} items {fo['pass']} of {fo['of']} {fn['pass']} of {fn['of']}"),
        ("num_batch", f"{T} num_batch: file naming time",
         rf"File naming time, median and 95th percentile {fo['p50_ms']} ms and {fo['p95_ms']} ms "
         rf"{fn['p50_ms']} ms and {fn['p95_ms']} ms"),
    ] + [("num_batch", f"{T} num_batch: first word on {name}",
          rf"First word on {re.escape(name)}, median {ft[k]['replaced']:.2f} s {ft[k]['grux_local']:.2f} s")
         for k, name in FIRST_WORD] + [
        ("num_batch", f"{T} num_batch: chat p95 on {name}",
         rf"Chat replies on {name}, 95% within {o['chat_p95_s']:.1f} s {n['chat_p95_s']:.1f} s")
        for name, o, n in (("held-out v1", o1, n1), ("held-out v2", o2, n2), ("the local turns at num_ctx 8,192", o8, n8))]
    # The sets where the two builds gave the same pass or fail on every task, named only when they did.
    same = c["holdout_v1"]["flips"] == c["holdout_v2"]["flips"] == c["s1"]["flips"] == 0
    rows.append(("num_batch", f"{T} num_batch: the same pass or fail on held-out v1, v2 and S1",
                 (rf"held-out v1 \({o1['of']} tasks\), held-out v2 \({o2['of']}\) and the S1 suite \({c['s1']['grux_local']['of']}\)"
                  r"[^.]{0,80}?the same pass or fail on every task") if same else r"(?!)"))
    for k in moved["lost"] + moved["gained"]:
        rows.append(("num_batch", f"{T} num_batch: case {k} named", rf"\b{re.escape(k)}\b"))
    rows.append(("num_batch", f"{T} num_batch: {len(moved['lost'])} lost and {len(moved['gained'])} gained at 8,192",
                 rf"\blost {WORDS[len(moved['lost'])]} hidden-instruction cases\b" if all("injection" in k for k in moved["lost"])
                 else r"(?!)"))
    rows.append(("num_batch", f"{T} num_batch: num_batch 128 lost the same cases",
                 rf"\bnum_batch 128\b[^.]{{0,80}}?\blost the same {WORDS[len(moved['lost'])]}\b"
                 if set(b128["lost"]) == set(moved["lost"]) else r"(?!)"))
    # At num_ctx 16,384 too (review 2026-11-27, P2: the card named only 8,192 while the 16,384 run lost the same cases).
    h16o = [r["hidden_instructions_ignored"] for r in o16]
    h16n = [r["hidden_instructions_ignored"] for r in n16]
    hc16 = n16[0]["hidden_instruction_cases"]
    rows.append(("num_batch", f"{T} num_batch: hidden instructions at 16,384, {span(h16o)} and {span(h16n)} of {hc16}",
                 rf"Of those at num_ctx 16,384, hidden instructions[^0-9]{{0,60}}? {span(h16o)} of {hc16}[^0-9]{{0,20}}? "
                 rf"{span(h16n)} of {hc16}"))
    m16 = c["local_turns"]["num_ctx_16384_against_first_run"]
    rows.append(("num_batch", f"{T} num_batch: the same cases lost at 16,384",
                 rf"\bat num_ctx 16,384\b[^.]{{0,160}}?\blost the same {WORDS[len(m16['lost'])]}\b"
                 if set(m16["lost"]) == set(moved["lost"]) else r"(?!)"))
    rows.append(("num_batch", f"{T} num_batch: the waiver", r"\bDotcomJack waived the rule for this build on November 27, 2026\b"))
    # Every place it is worse than the build it replaced, said as "new against old".
    worse = []
    if n8["pass"] < o8["pass"]:
        worse.append(rf"\b{n8['pass']} against {o8['pass']}\b")
    if n8["hidden_instructions_ignored"] < o8["hidden_instructions_ignored"]:
        worse.append(rf"\b{n8['hidden_instructions_ignored']} against {o8['hidden_instructions_ignored']} of {hc8}\b")
    if max(h16n) < min(h16o):
        worse.append(rf"\b{span(h16n)} against {span(h16o)} of {hc16}\b")
    worse += [rf"{ft[k]['grux_local']:.2f} s against {ft[k]['replaced']:.2f} s"
              for k, _ in FIRST_WORD if ft[k]["grux_local"] > ft[k]["replaced"]]
    worse += [rf"{n['chat_p95_s']:.1f} s against {o['chat_p95_s']:.1f} s"
              for o, n in ((o1, n1), (o2, n2), (o8, n8)) if float(f"{n['chat_p95_s']:.1f}") > float(f"{o['chat_p95_s']:.1f}")]
    worse += [rf"{fn[k]} ms against {fo[k]} ms" for k in ("p50_ms", "p95_ms") if fn[k] > fo[k]]
    if fn["pass"] < fo["pass"]:
        worse.append(rf"\b{fn['pass']} against {fo['pass']} of {fn['of']}\b")
    if gb_n > gb_o:
        worse.append(rf"{gb_n:.2f} GB against {gb_o:.2f} GB")
    fpo, fpn = c["runner_footprint_after_use_mb"]["replaced"], c["runner_footprint_after_use_mb"]["grux_local"]
    rows.append(("num_batch", f"{T} num_batch: runner footprint {fpo:,} and {fpn:,} MB",
                 rf"after 464 held-out v1 tasks {fpo:,} MB {fpn:,} MB"))
    if fpn > fpo:
        worse.append(rf"{fpn:,} MB against {fpo:,} MB")
    rows += [("worse", f"{T} num_batch is worse: {w}", w) for w in worse]
    return rows


def rebuilt_headline(d):
    """The Grux README's paragraph for a rebuilt tag: both held-out sets against the first build, every place it is
    worse, and, when only a parameter changed since, what that did."""
    a, b = d["grux_local"], d["previous_2b_same_machine"]
    a2, b2 = d["holdout_v2"]["grux_local"], d["holdout_v2"]["previous_2b_same_machine"]
    ha = span([r["families"]["chat_honesty"]["pass"] for r in d["local_turns"]["grux_local"]])
    hb = span([r["families"]["chat_honesty"]["pass"] for r in d["local_turns"]["previous_2b_same_machine"]])
    hof = d["local_turns"]["grux_local"][0]["families"]["chat_honesty"]["of"]
    fa, fb = d["file_naming"]["grux_local"], d["file_naming"]["previous_2b_same_machine"]
    T = d["size"]
    rows = [("headline", f"{T} rebuilt, both held-out sets",
             rf"\bit got {a2['pass']} of {a2['of']} on held-out v2 against {b2['pass']}, and {a['pass']} of {a['of']} "
             rf"on held-out v1 against {b['pass']}\b")]
    if a["hidden_instructions_ignored"] < b["hidden_instructions_ignored"]:
        rows.append(("worse", f"{T} rebuilt, fewer hidden instructions",
                     rf"\({a['hidden_instructions_ignored']} against {b['hidden_instructions_ignored']}\)"))
    rows.append(("worse" if ha != hb else "headline", f"{T} rebuilt, honesty on local turns",
                 rf"\({ha} against {hb} of\s+{hof}\)"))
    if fa["pass"] < fb["pass"]:
        rows.append(("worse", f"{T} rebuilt, file naming", rf"\({fa['pass']} against {fb['pass']} of {fa['of']}\)"))
    slow = [(n, x, y) for n, x, y in (("held-out v1", a, b), ("held-out v2", a2, b2)) if x["chat_p95_s"] > y["chat_p95_s"]]
    if len(slow) == 1:
        n, x, y = slow[0]
        rows.append(("worse", f"{T} rebuilt, chat speed on {n}",
                     rf"in chat speed on {n} \(95% within {x['chat_p95_s']:.1f} s against {y['chat_p95_s']:.1f} s\)"))
    elif slow:
        rows.append(("worse", f"{T} rebuilt, chat speed",
                     rf"95% within {a['chat_p95_s']:.1f} s against {b['chat_p95_s']:.1f} s on held-out v1, and "
                     rf"{a2['chat_p95_s']:.1f} s against {b2['chat_p95_s']:.1f} s on held-out v2\)"))
    c = d.get("num_batch_change")
    if c:
        k = "local_num_ctx_8192"
        o8, n8 = ([r for r in c["local_turns"][s] if r["num_ctx"] == 8192][0] for s in ("replaced", "grux_local"))
        rows.append(("headline", f"{T} num_batch line",
                     rf"\bnum_batch {d['params']['num_batch']}\b.{{0,200}}?from {c['first_token_median_s'][k]['replaced']:.2f} s "
                     rf"to {c['first_token_median_s'][k]['grux_local']:.2f} s\b.{{0,200}}?\({n8['pass']} against {o8['pass']}\)"))
    return rows


def blog_view(v1):
    """v1.json as it stood when the dated blog was written: each replaced build back in its tag."""
    old = json.loads(json.dumps(v1))
    for r in v1.get("replaced", []):
        old["tags"][r["tag"]] = {k: v for k, v in r.items() if k not in ("tag", "replaced_by", "loaded_after_use",
                                                                          "runtime", "input")}
        old["loaded_after_use"] = [{"tag": r["tag"], **r["loaded_after_use"]} if u["tag"] == r["tag"] else u
                                   for u in old["loaded_after_use"]]
    return old


def expected(v1):
    t = v1["tags"]
    rows, own, texts, rebuilt = [], [], [], []
    allowed = {"pairs": set(), "of_tasks": collections.defaultdict(set), "of_hidden": set(), "gb": set(), "s": set()}
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
        texts += text_only_rows(tag, d)
        if "previous_2b_same_machine" in d:
            rebuilt += rebuilt_rows(tag, d)
            for r in [r for _, r in runs_of(d, v1["machine"])]:
                allowed["pairs"].add((r["pass"], pct(r["pass"], r["of"])))
                allowed["of_tasks"][r["of"]].add(r["pass"])
                allowed["of_hidden"].add(r["hidden_instructions_ignored"])
                allowed["s"].add(f"{r['chat_p95_s']:.1f}")
            for k in ("grux_local", "previous_2b_same_machine"):
                for r in d["local_turns"][k]:
                    h = r["families"]["chat_honesty"]
                    allowed["of_tasks"][h["of"]].add(h["pass"])
                allowed["of_tasks"][d["file_naming"][k]["of"]].add(d["file_naming"][k]["pass"])
        if "num_batch_change" in d:   # the build it replaced, beside it on the card
            c = d["num_batch_change"]
            rebuilt += num_batch_rows(tag, d)
            for k in ("holdout_v1", "holdout_v2"):
                r = c[k]["replaced"]
                allowed["pairs"].add((r["pass"], pct(r["pass"], r["of"])))
                allowed["of_tasks"][r["of"]].add(r["pass"])
                allowed["of_hidden"].add(r["hidden_instructions_ignored"])
                allowed["s"].add(f"{r['chat_p95_s']:.1f}")
            allowed["gb"].add(f"{c['loaded_gb']['replaced']:.2f}")
            # "N of 67": the hidden instructions among the local turns, every run of every build named
            for r in c["local_turns"]["replaced"] + c["local_turns"]["grux_local"]:
                allowed["of_tasks"][r["hidden_instruction_cases"]].add(r["hidden_instructions_ignored"])
                if r["num_ctx"] == 8192:
                    allowed["s"].add(f"{r['chat_p95_s']:.1f}")
            allowed["of_tasks"][c["file_naming"]["replaced"]["of"]].add(c["file_naming"]["replaced"]["pass"])
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
        ] + (own_machine_headline(v2) if v2 else []) + texts + [
            r for tag, d in t.items() if "previous_2b_same_machine" in d for r in rebuilt_headline(d)],
        "site": [
            ("headline", "4B lead", rf"the 4B got {big['grux_local']['pass']} right, against "
                                    rf"{big['before_training']['pass']} for the same model before training"),
        ] + own + texts,
        # The model card carries the whole 4b-v2 section and the rebuilt 2B's; the Grux README one line (above).
        "card": own + texts + rebuilt,
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


def _raw(method, url, headers=None):
    """One request, no redirect followed, no body read: (status, lowercased headers). A blob body is never pulled."""
    u = urllib.parse.urlsplit(url)
    conn = (http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection)(u.netloc, timeout=30)
    try:
        conn.request(method, u.path + (f"?{u.query}" if u.query else ""), headers={**UA, **(headers or {})})
        r = conn.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}
    finally:
        conn.close()


def pull_path(registry, v1):
    """Ollama 0.40.8's pull, step by step, for every tag whose layers are not tensor blobs (a GGUF, like the 2b since
    2026-11-24). Such a pull goes through server/images.go downloadBlob, which (1) HEADs a layer with a same-host-only
    redirect policy (remoteBlobLayer, makeRequest): a cross-host redirect there is "blocked redirect to a different
    host", the error that failed every :2b pull for about 20 minutes on 2026-11-24; and (2) GETs it once to find a
    direct URL, then calls resp.Location(): an answer with no Location ("http: no Location header in response", what
    a preview host gives) fails too, and the Location must answer a ranged GET with 206. MLX tags (tensor layers) take
    the transfer path, which blob_burst covers. Not modelled: Ollama's private-address refusal for a redirect target
    (redirect.go ValidateRedirectTarget; our target is pages.dev) and its user agent (this sends UA, so a rule keyed
    on Ollama's agent would not show here). Returns the number of tags that failed."""
    base, failed = registry.rstrip("/"), 0
    host = urllib.parse.urlsplit(base).hostname
    for tag in v1["tags"]:
        url = f"{base}/v2/grux/grux-local/manifests/{tag}"
        try:
            body, _ = fetch(url, {"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
            m = json.loads(body)
            blobs = [m["config"]] + list(m["layers"])   # Ollama appends the config to the layers it pulls
            gguf = not any(l["mediaType"].endswith(".tensor") for l in m["layers"])
        except Exception as e:
            print(f"FAIL pullpath {tag}: no usable manifest ({e!r})")
            failed += 1
            continue
        if not gguf:
            continue
        problems, targets = [], set()
        for l in blobs:
            try:
                blob = f"{base}/v2/grux/grux-local/blobs/{l['digest']}"
                st, h = _raw("HEAD", blob)
                if st in (301, 302, 303, 307, 308) and urllib.parse.urlsplit(h.get("location", "")).hostname not in (None, host):
                    problems.append(f"{l['digest'][7:19]} HEAD {st} to {urllib.parse.urlsplit(h['location']).hostname} (Ollama blocks a cross-host redirect here)")
                elif st != 200 or h.get("content-length") != str(l["size"]):
                    problems.append(f"{l['digest'][7:19]} HEAD {st}, length {h.get('content-length')} (want {l['size']})")
                st, h = _raw("GET", blob)
                loc = h.get("location")
                if st not in (200, 301, 302, 303, 307, 308):   # download.go run(): any other status is an error
                    problems.append(f"{l['digest'][7:19]} GET {st} (Ollama accepts only 200 or a redirect here)")
                    continue
                if not loc:
                    problems.append(f"{l['digest'][7:19]} GET {st} with no Location (Ollama needs one to find the blob)")
                    continue
                loc = urllib.parse.urljoin(blob, loc)   # Go's resp.Location() resolves a relative one the same way
                if base.startswith("https://") and not loc.startswith("https://"):
                    problems.append(f"{l['digest'][7:19]} Location {loc[:60]} drops https (Ollama refuses the downgrade)")
                    continue
                st2, h2 = _raw("GET", loc, {"Range": "bytes=0-0"})
                if st2 != 206:
                    problems.append(f"{l['digest'][7:19]} Location answered {st2} to a ranged GET, not 206")
                targets.add(urllib.parse.urlsplit(loc).hostname)
            except Exception as e:
                problems.append(f"{str(l.get('digest', '(no digest)') if isinstance(l, dict) else l)[7:19] or l!r} error {e!r}")
        ok = not problems
        failed += not ok
        if ok:
            print(f"PASS pullpath {tag}: {len(m['layers']) + 1} blobs, each HEAD answered on {host}, each GET redirected "
                  f"to {', '.join(sorted(targets))} and answered 206 there")
        else:
            print(f"FAIL pullpath {tag}: {len(problems)} problems across {len(m['layers']) + 1} blobs on {host}")
        for p in problems[:8]:
            print(f"     {p}")
    return failed


def main():
    ap = argparse.ArgumentParser()
    for k, v in LIVE.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--results", default=os.path.join(HERE, "v1.json"))
    ap.add_argument("--blobs", type=int, default=0, help="ranged blob GETs to fire at once (0 = skip)")
    a = ap.parse_args()
    v1 = json.load(open(a.results))
    rows, headline, allowed, of, hc = expected(v1)
    old = blog_view(v1)
    _, headline_old, allowed_old, _, _ = expected(old)
    headline["blog"] = headline_old["blog"]
    scoped, problems = baselines(v1)
    scoped_old, problems_old = baselines(old)
    problems += [p for p in problems_old if p not in problems]
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
        if name == "readme" and not re.search(r"^> \*\*[^\n]*?Grux Local", md, re.M):
            print(f"FAIL readme: no Grux Local callout (a blockquote opening in bold that names Grux Local) in {src}")
            failed += 1
            continue
        ours = readme_scope(md).encode() if name == "readme" else raw
        text = text_of(ours, src)
        want = ([] if name == "blog" else rows) + headline.get(name, [])
        miss = [label for _, label, rx in want if not re.search(rx, text)]
        view = old if name == "blog" else v1
        n, bad = shape(text, allowed_old if name == "blog" else allowed, of, hc)
        claims = absolutes(text, view)
        n_scope, loose = unscoped(text, sentences_of(ours), table_spans(ours, text),
                                  scoped_old if name == "blog" else scoped, view)
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
    failed += pull_path(a.registry, v1)
    if a.blobs:
        failed += not blob_burst(a.registry, v1, a.blobs)
    print("ALL PASS" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
