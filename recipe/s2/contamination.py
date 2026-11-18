#!/usr/bin/env python3
"""Training cases that touch the frozen held-out sets, by id only (the orchestrator never reads either
side's text). A training case is excluded when its user-facing text equals a held-out case's, or shares
any 8-word run with one. The held-out list is version 1 (s2/holdout.py), version 2 (s2/holdout_v2.py) and
version 3 (s2/holdout_v3.py), the three the swing charter names.
Writes s2/train_exclude.json; gen_data.py skips those ids. The v3-shaped bank (s2/train_bank_v3.py) is checked by
v3_excluded, every text of a job one at a time, and never lands in train_exclude.json (its ids are its own).

  python3 s2/contamination.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import holdout  # noqa: E402
import holdout_v2  # noqa: E402
import holdout_v3  # noqa: E402
import train_bank  # noqa: E402

SKIP = {"id", "family", "now", "tool", "args", "expect_keys"}


def texts(c):
    out = []
    for k, v in c.items():
        if k in SKIP:
            continue
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out += [x for x in v if isinstance(x, str)]
    return re.findall(r"[a-z0-9']+", " ".join(out).lower())


def grams(ws, n=8):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def held_out():
    """The word lists every training case is checked against: each held-out v1 and v2 case's text, and each text a
    held-out v3 job shows a model (holdout_v3.case_texts: the ask, the history, every scripted call and result)."""
    v3 = [re.findall(r"[a-z0-9']+", t.lower()) for c in holdout_v3.cases() for t in holdout_v3.case_texts(c)]
    return ([texts(c) for c in holdout.cases()] + [texts(c) for c in holdout_v2.cases()]
            + [w for w in v3 if len(w) >= 3])   # not "Got it." as a whole text


# Lines Grux itself prints at holdout_v3.GRUX_TAG, whole, with only counts and a day as slots: (the line, the source
# file, the literal pieces of it that file holds, which test_train_bank_v3 checks). A v3-shaped training case carries
# them in a tool result exactly as held-out v3 does, so they are product output, not held-out content. Only a whole
# line that matches one of them is set aside; every other line, and every word around it, is still checked.
RUNTIME_LINES = [
    (r"ok: \d+ snapshots? from the last \d+ days(?:, showing \d+)?, the ones taken before a destructive command "
     r"first, then newest first\.", "ShellTool/ShellRecoveryTools.swift",
     (" from the last ", ", the ones taken before a destructive command first, then newest first.")),
    (r"ok: no snapshot from the last \d+ days can be restored from chat\.", "ShellTool/ShellRecoveryTools.swift",
     ("ok: no snapshot from the last ", " days can be restored from chat.")),
    (r"pending: this is waiting for your OK in Approvals\. Nothing is sent or changed until you approve it\.",
     "Jax/JaxToolGate.swift",
     ("pending: this is waiting for your OK in Approvals.", " Nothing is sent or changed until you approve it.")),
    (r"ok: \d+ lines? (?:today|yesterday|[A-Z][a-z]+day)(?: with outcome refused)?, \d+ shell commands? ran ok "
     r"(?:today|yesterday|[A-Z][a-z]+day)\.", "ShellTool/ShellRecoveryTools.swift",
     (" with outcome ", " shell command", " ran ok ")),
]
_RUNTIME = re.compile("|".join(f"(?:{p})" for p, _, _ in RUNTIME_LINES))


def v3_texts(c):
    """Every text a held-out-v3-shaped case shows a model (holdout_v3.case_texts: the ask, each earlier turn, each
    scripted call and result), one per message, with RUNTIME_LINES set aside line by line."""
    out = []
    for t in holdout_v3.case_texts(c):
        t = "\n".join(line for line in t.split("\n") if not _RUNTIME.fullmatch(line.strip()))
        if t.strip():
            out.append(t)
    return out


def v3_excluded(cases):
    """{id: why} for each v3-shaped training case (s2/train_bank_v3.py) that shares a whole text (3 words or more)
    or any 8-word run with held-out v1, v2 or v3, one text at a time, by held_out()'s reading of those sets."""
    held = held_out()
    whole = {" ".join(w) for w in held if len(w) >= 3}
    runs = set().union(*[grams(w) for w in held])
    memo, out = {}, {}
    for c in cases:
        for t in v3_texts(c):
            if t not in memo:
                ws = re.findall(r"[a-z0-9']+", t.lower())
                joined = " ".join(ws)
                shared = sorted(grams(ws) & runs)
                memo[t] = ("same text: " + joined[:60] if len(ws) >= 3 and joined in whole
                           else shared[0] if shared else None)
            if memo[t]:
                out[c["id"]] = memo[t]
                break
    return out


# --------------------------------------------------------------------------- the template-clone gate
# The 8-word rule cannot see "same template, other slot values". skeleton() strips the slot values (anything in
# backticks or quotes, URLs, addresses and paths, times, amounts and numbers, day and month names, and capitalized
# words inside a sentence, which are the names), so two lines from one template read the same; near_v3 scores a
# line by its highest word-4-gram Jaccard against any held-out v3 text, and a line too short for a 4-gram by an
# exact match of its whole skeleton. test_train_bank_v3 sets the threshold.
# What it does not see: a lowercase noun swapped for another ("a pizza stone" for "a kiln") keeps the shape but
# changes the words, so the gate proves there is no near-identical clone of a held-out line, not that no line
# shares a held-out line's idea. That second guard is the rewrite itself, read by a reviewer.

_DAYS = ("monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|"
         "august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|today|"
         "tomorrow|yesterday|tonight")
_SLOTS = [(re.compile(r"`[^`]*`"), " C "),
          (re.compile(r"(?<![A-Za-z])['\"‘“][^'\"’”\n]{2,80}['\"’”](?![A-Za-z])"), " Q "),
          (re.compile(r"https?://\S+|\S+@\S+\.\S+|~?/[\w./*-]+|\b[\w-]+\.(?:example|com|org|net|gov|be)\S*"), " U "),
          (re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)(?![a-z])|\b\d{1,2}:\d{2}\b"), " T "),
          (re.compile(r"\$[\d,.]+|\b\d[\d,.]*\b"), " # ")]
_DAYS_RE = re.compile(rf"(?i)\b(?:{_DAYS})\b")
_NAME = re.compile(r"(?<=[A-Za-z0-9,;)'’] )[A-Z][A-Za-z'’-]*")   # a capital that is not the first word of a sentence


def skeleton(text):
    """The words of text with every slot value replaced by a placeholder, runs of one placeholder collapsed."""
    t = text
    for rx, ph in _SLOTS:
        t = rx.sub(ph, t)
    t = _DAYS_RE.sub(" D ", t)
    t = _NAME.sub("N", t)
    out = []
    for w in re.findall(r"[A-Za-z']+|[#]", t):
        w = w if w in ("C", "Q", "U", "T", "D", "N", "#") else w.lower()
        if not (out and w == out[-1] and w in ("C", "Q", "U", "T", "D", "N", "#")):
            out.append(w)
    return out


def grams4(ws):
    return {" ".join(ws[i:i + 4]) for i in range(len(ws) - 3)}


def v3_reference():
    """[(case id, skeleton 4-grams, whole skeleton)] for every held-out v3 text a model reads, one per text."""
    out = []
    for c in holdout_v3.cases():
        for t in holdout_v3.case_texts(c):
            sk = skeleton(t)
            if sk:
                out.append((c["id"], grams4(sk), " ".join(sk)))
    return out


def near_v3(lines, ref=None, skip_case=None):
    """{line: (highest Jaccard, held-out case id)} over held-out v3's texts (v3_reference). skip_case(line) names a
    held-out case id whose texts are left out (for scoring held-out's own lines against the rest of it)."""
    ref = v3_reference() if ref is None else ref
    index, shorts = {}, {}
    for i, (cid, g, whole) in enumerate(ref):
        for x in g:
            index.setdefault(x, []).append(i)
        if not g:
            shorts.setdefault(whole, []).append(cid)
    out = {}
    for line in lines:
        sk = skeleton(line)
        g = grams4(sk)
        best, who = 0.0, None
        own = skip_case(line) if skip_case else None
        if not g:   # under 4 words: the same short skeleton as a held-out text is a clone
            who = next((cid for cid in shorts.get(" ".join(sk), ()) if cid != own), None)
            best = 1.0 if who else 0.0
        if g:
            hits = {}
            for x in g:
                for i in index.get(x, ()):
                    hits[i] = hits.get(i, 0) + 1
            for i, n in hits.items():
                if ref[i][0] == own:
                    continue
                j = n / len(g | ref[i][1])
                if j > best:
                    best, who = j, ref[i][0]
        out[line] = (best, who)
    return out


SHORT = 12   # a line under this many skeleton words also gets the short-line score


def grams3(ws):
    return {" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)}


def near_v3_short(lines, skip_case=None, ref=None):
    """{line: (highest containment, held-out case id)} for each line under SHORT skeleton words: the share of the
    line's word-3-grams found in one held-out v3 text. A one-word swap in a six-word line keeps half of its 3-grams
    and scores 0.5, where its 4-gram Jaccard reads 0.20. Longer lines score (0.0, None). `ref` ([(id, 3-grams)])
    replaces held-out v3's texts (v4_clones passes every set v4 is gated against)."""
    if ref is None:
        ref = []
        for c in holdout_v3.cases():
            for t in holdout_v3.case_texts(c):
                g = grams3(skeleton(t))
                if g:
                    ref.append((c["id"], g))
    index = {}
    for i, (_, g) in enumerate(ref):
        for x in g:
            index.setdefault(x, []).append(i)
    out = {}
    for line in lines:
        sk = skeleton(line)
        g = grams3(sk)
        best, who = 0.0, None
        if g and len(sk) < SHORT:
            own = skip_case(line) if skip_case else None
            hits = {}
            for x in g:
                for i in index.get(x, ()):
                    hits[i] = hits.get(i, 0) + 1
            for i, n in hits.items():
                if ref[i][0] != own and n / len(g) > best:
                    best, who = n / len(g), ref[i][0]
        out[line] = (best, who)
    return out


# --------------------------------------------------------------------------- held-out v4's gate
# Held-out v4 (s2/holdout_v4.py) shares nothing with held-out v1, v2 or v3, with the v3 training bank (every case
# s2/train_bank_v3.py generates, before its exclusion) or with the S2 bank (every case of s2/train_bank.py): no whole
# text of 3 words or more, no 8-word run, and no line of its own words (an ask, a planted line, an earlier turn) that
# is a template clone of one of theirs by the two scores above. Grux's own output lines are product output, not
# held-out content: RUNTIME_LINES plus V4_RUNTIME_LINES are set aside in v4's texts line by line, as v3_texts does for
# the bank. Nothing here changes what v3_excluded or main() read.

V4_RUNTIME_LINES = RUNTIME_LINES + [
    (r"pending: your email to \S+ is waiting for your OK in Approvals\. It would go out as [A-Z][a-z]+\. Nothing is "
     r"sent until you approve it\.", "Email/Imap/EmailTool.swift",
     ("pending: your email to ", " is waiting for your OK in Approvals. ", "It would go out as ",
      ". Nothing is sent until you approve it.")),
]
_RUNTIME_V4 = re.compile("|".join(f"(?:{p})" for p, _, _ in V4_RUNTIME_LINES))
# The template-clone thresholds test_train_bank_v3 set from held-out v3's own spread (its check 8), unchanged.
V4_PARA_MAX = 0.25
V4_SHORT_MAX = 0.5


def _words(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def v4_texts(c):
    """Every text a held-out-v4 case shows a model (holdout_v3.case_texts), one per message, with Grux's own output
    lines (V4_RUNTIME_LINES) set aside line by line."""
    out = []
    for t in holdout_v3.case_texts(c):
        t = "\n".join(line for line in t.split("\n") if not _RUNTIME_V4.fullmatch(line.strip()))
        if t.strip():
            out.append(t)
    return out


def _fields(c):
    """Each string an S2-shaped case holds (the fields texts() reads), one at a time."""
    out = []
    for k, v in c.items():
        if k in SKIP:
            continue
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out += [x for x in v if isinstance(x, str)]
    return out


def v4_reference():
    """{set name: (word lists for the 8-word rule, strings for whole-text and clone checks)} for every set held-out
    v4 must not touch. v1, v2 and the S2 bank are read as texts() reads them (one word list per case, so a run that
    crosses two fields counts) and field by field; v3 and the v3 bank one text per message (holdout_v3.case_texts)."""
    import train_bank_v3
    out = {}
    for name, cs in (("held-out v1", holdout.cases()), ("held-out v2", holdout_v2.cases()),
                     ("S2 training bank", train_bank.cases())):
        lines = [s for c in cs for s in _fields(c)]
        out[name] = ([texts(c) for c in cs] + [_words(s) for s in lines], lines)
    for name, cs in (("held-out v3", holdout_v3.cases()), ("v3 training bank", train_bank_v3.cases())):
        lines = [t for c in cs for t in holdout_v3.case_texts(c)]
        out[name] = ([_words(t) for t in lines], lines)
    return out


def v4_overlap(cases, ref=None):
    """{case id: '<set>: <first shared 8-word run>' or '<set>: same text: ...'} for each held-out v4 case that shares
    a whole text of 3 words or more, or any 8-word run, with a set in v4_reference."""
    ref = v4_reference() if ref is None else ref
    index = []
    for name, (wls, _) in ref.items():
        runs, whole = set(), set()
        for ws in wls:
            runs |= grams(ws)
            if len(ws) >= 3:
                whole.add(" ".join(ws))
        index.append((name, runs, whole))
    out = {}
    for c in cases:
        for t in v4_texts(c):
            ws = _words(t)
            joined, g = " ".join(ws), grams(ws)
            hit = next((f"{name}: same text: {joined[:60]}" if len(ws) >= 3 and joined in whole
                        else f"{name}: {sorted(g & runs)[0]}"
                        for name, runs, whole in index if (len(ws) >= 3 and joined in whole) or g & runs), None)
            if hit:
                out[c["id"]] = hit
                break
    return out


def v4_clone_ref(ref=None):
    """(4-gram reference for near_v3, 3-gram reference for near_v3_short) over every string of every set in
    v4_reference, one entry per distinct skeleton, named by its set."""
    ref = v4_reference() if ref is None else ref
    seen, four, three = set(), [], []
    for name, (_, lines) in ref.items():
        for t in lines:
            sk = skeleton(t)
            whole = " ".join(sk)
            if not sk or whole in seen:
                continue
            seen.add(whole)
            four.append((name, grams4(sk), whole))
            g3 = grams3(sk)
            if g3:
                three.append((name, g3))
    return four, three


def v4_clones(lines, ref=None):
    """{line: (score, set, which score)} for each line of held-out v4's own words that reads as a template clone of
    a line in v4_reference: 4-gram Jaccard at or above V4_PARA_MAX, or, for a line under SHORT skeleton words, 3-gram
    containment at or above V4_SHORT_MAX."""
    four, three = v4_clone_ref(ref)
    out = {}
    long_ = near_v3(lines, four)
    short = near_v3_short(lines, ref=three)
    for line in lines:
        v4_, who4 = long_[line]
        v3_, who3 = short[line]
        if v4_ >= V4_PARA_MAX:
            out[line] = (round(v4_, 3), who4, "4-gram Jaccard")
        elif v3_ >= V4_SHORT_MAX:
            out[line] = (round(v3_, 3), who3, "3-gram containment")
    return out


# The frame gate. A line whose nouns all change keeps few 4-grams, so near_v3 cannot see it ("Read my document Dance
# recital and put the dress rehearsal on my calendar" against held-out v3's "Read my document Field trip and put the
# trip on my calendar" scores 0.06). frame() keeps only the closed-class words of a skeleton (pronouns, prepositions,
# conjunctions, auxiliaries, question words, common contractions) and its slot placeholders, drops the articles, and
# masks every other word as X, one X per run; the two lines above both read "X my X N X and X X on my X". A line is a
# frame echo when its frame is within FRAME_EDITS token edits of a reference line's frame (an X inserted or deleted is
# free; FRAME_EDITS_LONG when both frames have FRAME_LONG tokens or more), or its frame 3-gram Jaccard is V4_FRAME_MAX or more; lines with fewer than FRAME_FIXED
# fixed words are not frame-scored. Edits are counted on the fixed words alone (_fixed_seq), and a reference is found
# by a piece of them (_chunks), so masked words between them never hide it. Measured on the Mini over every
# closed-class position of held-out v3's asks: every probe with one word swapped, made an open-class word or dropped
# that keeps FRAME_FIXED fixed words is caught (test_holdout_v4 asserts it); Jaccard alone caught 107 of 164 one-word
# swaps. Turning the edit rule on flagged 11 v4 lines of 592 before their rewrite.
FRAME_WORDS = frozenset("""my your our their his her its this that these those it i me we us you he she they them
to of in on at for from with by about into onto over under after before until up down out off as than then and or
but if so not no is are was were be been am do does did have has had can could will would should may might must
what when where which who whom whose why how there here all any some every each one two please just also without
during through between next last now again more other another only still yet too very both either neither nor once
while because since across along around near per via against among below above behind beside within don't can't
won't isn't aren't didn't doesn't haven't hasn't wouldn't couldn't shouldn't what's it's that's there's i'm i'll i'd
i've you're we're they're let's he's she's here's where's who's how's""".split())
FRAME_FIXED = 5
FRAME_EDITS, FRAME_EDITS_LONG, FRAME_LONG = 1, 2, 10
V4_FRAME_MAX = 0.5
_PLACEHOLDERS = frozenset({"C", "Q", "U", "T", "D", "N", "#"})


def frame(text):
    """The skeleton of text (curly apostrophes made straight) with the articles dropped and every word that is not a
    closed-class word or a slot placeholder masked as X, a run of X kept as one."""
    out = []
    for w in skeleton(text.replace(chr(0x2019), "'").replace(chr(0x2018), "'")):
        if w in ("a", "an", "the"):
            continue
        w = w if (w in _PLACEHOLDERS or w in FRAME_WORDS) else "X"
        if not (out and w == "X" == out[-1]):
            out.append(w)
    return out


def _fixed(f):
    return sum(w != "X" for w in f)


def _edits(a, b, cap):
    """Token edit distance between a and b, or cap + 1 once it is past cap."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _fixed_seq(f):
    """A frame's fixed words in order, the X left out: two frames' edit distance with an X inserted or deleted for
    free (a dropped word between two masked words merges two X into one, one change and not two) is the plain edit
    distance of these sequences."""
    return tuple(w for w in f if w != "X")


def _chunks(seq, k):
    """seq cut into k contiguous pieces as even as can be. Two sequences within k - 1 edits share one piece intact
    (an edit touches one piece at most), so a reference is a candidate when it holds one of them."""
    n, out, at = len(seq), [], 0
    for i in range(k):
        size = n // k + (1 if i < n % k else 0)
        out.append(seq[at:at + size])
        at += size
    return out


def v4_frame_ref(ref=None):
    """[(frame, set name, the line)] for every line, and every line of a multi-line text, in v4_reference whose frame
    has FRAME_FIXED fixed words, one entry per distinct frame."""
    ref = v4_reference() if ref is None else ref
    out, seen = [], set()
    for name, (_, lines) in ref.items():
        for t in lines:
            for x in [t] + [y.strip() for y in t.split("\n") if y.strip()]:
                f = tuple(frame(x))
                if _fixed(f) >= FRAME_FIXED and f not in seen:
                    seen.add(f)
                    out.append((f, name, x))
    return out


def v4_frames(lines, ref=None):
    """{line: (why, set name, the reference line)} for each line whose frame is within FRAME_EDITS edits of a
    reference frame on their fixed words (FRAME_EDITS_LONG when both frames have FRAME_LONG tokens), or whose frame
    3-gram Jaccard is V4_FRAME_MAX or more. Lines under FRAME_FIXED fixed words are not scored."""
    fr = v4_frame_ref(ref)
    seqs = [_fixed_seq(f) for f, _, _ in fr]
    by1, by2, by3, g3 = {}, {}, {}, []
    for i, (f, _, _) in enumerate(fr):
        sq = seqs[i]
        for w in set(sq):
            by1.setdefault(w, []).append(i)
        for x in {sq[k:k + 2] for k in range(len(sq) - 1)}:
            by2.setdefault(x, []).append(i)
        g = grams3(list(f))
        g3.append(g)
        for x in g:
            by3.setdefault(x, []).append(i)
    out = {}
    for line in lines:
        f = frame(line)
        if _fixed(f) < FRAME_FIXED:
            continue
        sq, mine = _fixed_seq(f), grams3(f)
        top = FRAME_EDITS_LONG if len(f) >= FRAME_LONG else FRAME_EDITS
        cands = set()
        for piece in _chunks(sq, top + 1):   # every reference within `top` edits holds one piece intact
            hits = by2.get(piece[:2], ()) if len(piece) >= 2 else by1.get(piece[0], ()) if piece else range(len(fr))
            cands.update(i for i in hits if any(seqs[i][k:k + len(piece)] == piece
                                                for k in range(len(seqs[i]) - len(piece) + 1)))
        for x in mine:   # every reference with a frame 3-gram Jaccard above 0
            cands.update(by3.get(x, ()))
        best = None
        for i in sorted(cands):
            cap = FRAME_EDITS_LONG if min(len(f), len(fr[i][0])) >= FRAME_LONG else FRAME_EDITS
            d = _edits(sq, seqs[i], cap)
            j = len(mine & g3[i]) / len(mine | g3[i]) if mine | g3[i] else 0.0
            if d <= cap or j >= V4_FRAME_MAX:
                why = f"{d} edit{'s' if d != 1 else ''}" if d <= cap else f"3-gram Jaccard {j:.2f}"
                if best is None or (d, -j) < best[0]:
                    best = ((d, -j), why, i)
        if best:
            out[line] = (best[1], fr[best[2]][1], fr[best[2]][2])
    return out


def main():
    held = held_out()
    whole = {" ".join(w) for w in held}
    runs = set().union(*[grams(w) for w in held])
    out = sorted(c["id"] for c in train_bank.cases()
                 if " ".join(texts(c)) in whole or grams(texts(c)) & runs)
    json.dump({"rule": "exact text or any shared 8-word run with s2/holdout.py (sha 1623e0e7), s2/holdout_v2.py "
                       f"(file sha {holdout_v2.sha()[:8]}) or s2/holdout_v3.py "
                       f"(case sha {holdout_v3.FROZEN_SHA[:8]})",
               "excluded": out}, open(os.path.join(HERE, "train_exclude.json"), "w"), indent=1)
    print(f"EXCLUDED {len(out)} training cases")


if __name__ == "__main__":
    main()
