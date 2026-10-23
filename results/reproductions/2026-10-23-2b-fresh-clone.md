# Reproduction: the 2B held-out score from a fresh clone (2026-10-23)

Who and where: Grux, on a 16 GB Mac mini (Apple M4) with Ollama 0.40.8, on October 23, 2026, starting from a
fresh clone of this repository at commit 1b7286c and following `recipe/README.md` step 1 word for word:

```
git clone https://github.com/dotcomjack/grux-local.git && cd grux-local
export GRUX_SRC="$(sh recipe/get_grux_source.sh)"
python3 recipe/check_prompts.py
ollama pull gruxai.com/grux/grux-local:2b
python3 recipe/eval_local.py --model gruxai.com/grux/grux-local:2b --set holdout --run /tmp/grux-local-check
python3 recipe/summarize.py /tmp/grux-local-check/*-holdout.jsonl
```

Output, verbatim:

```
464 held-out requests, hash 7c3f2a9ff3316ecc, expected 7c3f2a9ff3316ecc: MATCH
eval exit 0 in 446 s
pass 435/464 | hidden instructions ignored 66/70 | chat p95 2.1 s | unscored 0
{'ambient': '19/20', 'chat_honesty': '29/32', 'chat_plain': '40/41', 'chat_tools': '104/120', 'dictation': '48/51', 'email_audit': '28/28', 'email_triage': '26/26', 'injection': '66/70', 'issue': '15/15', 'meeting': '9/10', 'notification': '35/35', 'title': '16/16'}
```

Every count matches `results/v1.json` for the 2B (435 of 464, 66 of 70 hidden instructions, and each kind of task).
Chat timing differed from v1 (2.1 s against 1.6 s) because the Mac mini was running other work at the time; timing
depends on the Mac and its load, the counts do not.
