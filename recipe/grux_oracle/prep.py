"""Copy the Swift files the oracle needs, with each function that reads live app state (its header or body names
one of DROP) cut out, brace for brace. Matchers and pure functions are copied byte for byte."""
import os
import re
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
FILES = sys.argv[3:]
DROP = re.compile(r"\b(AppState|KeychainStore|ModelRegistry|SettingsPane|Cookbook|CookbookModel|ComposerFooter|"
                  r"ClaudeSwitchOffer|OllamaManager|PresetStore|CustomEndpointStore|WebResearch|SpeechEngine|"
                  r"FeatureRegistry|HowItWorksCopy|TuningCopy|UserIdentity|ChatReadiness|ProfileMemoryStore|JaxProfile|ClaudeMessage)\b")
HEAD = re.compile(r"\s*(@MainActor\s+)?(nonisolated\s+|private\s+|fileprivate\s+)*static (func|var) ")


def cut(text):
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if HEAD.match(line) and "{" in "".join(lines[i:i + 8]):
            j, depth, started = i, 0, False
            while j < len(lines):
                l = lines[j]
                depth += l.count("{") - l.count("}")
                started = started or "{" in l
                j += 1
                if started and depth <= 0:
                    break
            if DROP.search("\n".join(lines[i:j])):
                if out and out[-1].strip() == "@MainActor":
                    out.pop()
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


os.makedirs(OUT, exist_ok=True)
for f in FILES:
    text = open(os.path.join(SRC, f), encoding="utf-8").read()
    open(os.path.join(OUT, os.path.basename(f)), "w", encoding="utf-8").write(cut(text))
