#!/usr/bin/env python3
"""Refuse a training file that carries a secret or an identity string, before it goes into a box.

Two scans, over every string in every row:
  - key shapes: sk-..., ghp_/gho_/github_pat_, AKIA..., xox?-..., -----BEGIN ... KEY-----, and long hex (40+) or
    base64 (48+, mixed case and digits) runs;
  - identity terms: whole-word, case-insensitive, read from the vocabulary files named in GRUX_DATA_SCAN_VOCAB
    (colon-separated; JSON with a "banned" list, or one term per line; a path starting with ? is optional). The terms
    live outside this public repo.
A finding names the row id and the kind, never the matched text, so the log itself leaks nothing.

  GRUX_DATA_SCAN_VOCAB=<file>[:<file>] python3 recipe/s2/data_scan.py <train.jsonl>     exit 0 clean, 1 hits, 2 cannot scan
"""
import json
import os
import re
import sys

KEY_SHAPES = [
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("stripe key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}")),
    ("aws access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("long hex secret", re.compile(r"\b[0-9a-fA-F]{40,}\b")),
    ("long base64 secret", re.compile(r"(?<![A-Za-z0-9+/])(?=[A-Za-z0-9+/]*[a-z])(?=[A-Za-z0-9+/]*[A-Z])"
                                      r"(?=[A-Za-z0-9+/]*[0-9])[A-Za-z0-9+/]{48,}={0,2}")),
]


def load_terms(spec):
    """Terms from each file in the colon-separated spec. Raises when a named file cannot be read (fail closed)."""
    terms = []
    for path in [p for p in (spec or "").split(":") if p.strip()]:
        optional = path.startswith("?")
        path = os.path.expanduser(path.lstrip("?"))
        if optional and not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8-sig").read()
        if path.endswith(".json"):
            banned = json.loads(text)["banned"]
            if not isinstance(banned, list) or not all(isinstance(t, str) for t in banned):
                raise ValueError(f"{path}: 'banned' must be a list of strings")
            terms += [t for t in banned if t.strip()]
        else:
            terms += [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]
    return terms


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


def matcher(terms):
    """Whole-word, case-insensitive; any run of whitespace matches a space inside a term."""
    if not terms:
        return None
    alts = "|".join(r"\s+".join(re.escape(w) for w in t.split()) for t in terms)
    return re.compile(r"(?<![A-Za-z0-9])(?:" + alts + r")(?![A-Za-z0-9])", re.I)


def row_hit(row, word):
    """The kind of the first hit in one row (keys and values), or None. Never the matched text."""
    for s in strings(row):
        kind = next((k for k, rx in KEY_SHAPES if rx.search(s)), None)
        if kind is None and word is not None and word.search(s):
            kind = "identity term"
        if kind:
            return kind
    return None


def scan(path, terms):
    """[(row id, kind, line number)] for every row with a hit. A line that is not JSON is a hit ("unreadable row"):
    what cannot be read cannot be cleared."""
    word = matcher(terms)
    hits = []
    for n, raw in enumerate(open(path, "rb"), 1):
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            hits.append((f"line {n}", "unreadable row", n))
            continue
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            hits.append((f"line {n}", "unreadable row", n))
            continue
        rid = row.get("id", f"line {n}") if isinstance(row, dict) else f"line {n}"
        kind = row_hit(row, word)
        if kind:
            hits.append((rid, kind, n))
    return hits


def check(path, vocab_spec, require_vocab=True):
    """(ok, message). Fails closed: no readable vocabulary when one is required is a refusal, never a pass."""
    try:
        terms = load_terms(vocab_spec)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        return False, f"cannot read the identity vocabulary ({type(e).__name__}); refusing to upload unscanned data"
    if require_vocab and not terms:
        return False, "no identity vocabulary configured (GRUX_DATA_SCAN_VOCAB); refusing to upload unscanned data"
    try:
        hits = scan(path, terms)
        rows = sum(1 for l in open(path, "rb") if l.strip())
    except OSError as e:
        return False, f"cannot read the training data ({type(e).__name__}); nothing to upload"
    if rows == 0:
        return False, "the training data has no rows; refusing to train on nothing"
    if hits:
        kinds = sorted({k for _, k, _ in hits})
        return False, (f"{len(hits)} training rows carry a secret shape or an identity term ({', '.join(kinds)}); "
                       f"first ids {[r for r, _, _ in hits[:5]]}")
    return True, f"data scan clean: {rows} rows, {len(terms)} identity terms, {len(KEY_SHAPES)} key shapes"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    ok, msg = check(sys.argv[1], os.environ.get("GRUX_DATA_SCAN_VOCAB", ""))
    print(("CLEAN: " if ok else "REFUSED: ") + msg)
    sys.exit(0 if ok else (2 if "cannot read" in msg or "no identity" in msg else 1))
