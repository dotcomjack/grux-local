"""Grux 3.1.0's local chat turn, ported from the Swift source, for the `realistic` request mode.

suite.py offers every chat case the same 12 tools. Grux 3.1.0 does not: a local turn takes the whole
catalogue (`ChatService.allTools()`, 116 tools in a fresh install), keeps the groups the message calls
for (`ToolRelevance.filter`), ranks what is left by the message (`LocalTurn.rank`), offers as many as fit
the window beside the prompt and the reply (`LocalTurn.fit`), and names the rest in one line
(`LocalTurn.cutToolsLine`). This module reads the 116 schemas out of the Swift source the way suite.py
reads the prompts (no model, no build), ports each rule with the file and line it came from, and builds
the request a 3.1.0 local turn sends. Lines cite Grux tag v3.1.0 (commit 72d8549).

What a realistic request changes against suite.request, and nothing else:
  - tools: the ported filter, rank and fit over the 116 schemas, instead of suite.TOOLS;
  - the cut line, as a per-turn block, so it opens the user's message after the volatile block
    (ChatService.swift:762-764 appends it as an uncached system block; OpenAICompatBackend.swift:210-233
    moves uncached blocks to the head of the last user message on Ollama);
  - num_predict 1024 (LocalTurn.replyTokens, LocalTurn.swift:57; ChatService.swift:775).
The system prompt, the volatile block, temperature and num_ctx stay byte-identical to suite.request.

Known fidelity limit: suite.chat_system() leaves out the per-install WHAT_YOU_CAN_DO block, the profile and
the Jax identity body that the real lean prompt carries (ChatService.swift:2569). A longer system prompt
leaves less room for tools, so a real install offers at most what this mode offers.

  python3 local_turn.py          # self-test: catalogue size, the 12 suite tools byte for byte, the Swift
                                 # tests' own rank and fit cases, then a table of what each held-out turn gets
"""
import functools
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suite  # noqa: E402

PROMPT_CEILING = 8_192      # LocalTurn.promptCeiling, LocalTurn.swift:54
REPLY_TOKENS = 1_024        # LocalTurn.replyTokens, LocalTurn.swift:57
MARGIN = 256                # LocalTurn.fit, LocalTurn.swift:111
EVERYDAY = ["list_tasks", "add_task", "complete_task", "list_events", "create_event",
            "search_web", "capture_memory", "open_app"]          # LocalTurn.everyday, LocalTurn.swift:82-83


def context_tokens(memory_gb):
    """LocalTurn.contextTokens(memoryGB:), LocalTurn.swift:21-27: num_ctx by this Mac's memory."""
    return 8_192 if memory_gb < 12 else 16_384 if memory_gb < 24 else 32_768


def fit_window(memory_gb):
    """The window fit() is given: min(contextTokens, promptCeiling), ChatService.swift:760. So an 8 GB and a
    16 GB Mac fit the same tools; memory changes only num_ctx (OpenAICompatBackend.swift:43-49)."""
    return min(context_tokens(memory_gb), PROMPT_CEILING)


# --------------------------------------------------------------------------- Swift sizes

def swift_count(s):
    """Swift String.count: extended grapheme clusters. Enough of UAX 29 for prompt text: CR LF, combining
    marks, variation selectors, emoji modifiers, ZWJ sequences and regional-indicator pairs."""
    n, i, L = 0, 0, len(s)
    while i < L:
        c = s[i]
        i += 1
        if c == "\r" and i < L and s[i] == "\n":
            i += 1
        elif 0x1F1E6 <= ord(c) <= 0x1F1FF and i < L and 0x1F1E6 <= ord(s[i]) <= 0x1F1FF:
            i += 1
        while i < L:
            o = ord(s[i])
            if unicodedata.category(s[i]) in ("Mn", "Me", "Mc") or 0xFE00 <= o <= 0xFE0F or 0x1F3FB <= o <= 0x1F3FF \
                    or 0xE0020 <= o <= 0xE007F:
                i += 1
            elif o == 0x200D and i + 1 < L:
                i += 2
            else:
                break
        n += 1
    return n


def json_bytes(obj):
    """Byte length of JSONSerialization.data(withJSONObject:) on Darwin: compact, UTF-8, '/' escaped."""
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("/", "\\/").encode("utf-8"))


def tokens(chars):
    """LocalTurn.tokens(chars:), LocalTurn.swift:61."""
    return (chars * 2 + 6) // 7


def tool_chars(t):
    """LocalTurn.chars(_ t: ClaudeTool), LocalTurn.swift:141-143."""
    return swift_count(t["name"]) + swift_count(t["description"]) + json_bytes(t["schema"]) + 40


def message_chars(m):
    """LocalTurn.chars(_ m: [String: Any]), LocalTurn.swift:146-148."""
    return json_bytes(m)


# --------------------------------------------------------------------------- the catalogue, from source

def _strip_comments(src):
    """Swift source with // and /* */ comments blanked, string literals untouched."""
    out, i, L = [], 0, len(src)
    while i < L:
        if src.startswith('"""', i):
            j = src.index('"""', i + 3)
            while src[j - 1] == "\\" and src[j - 2] != "\\":
                j = src.index('"""', j + 1)
            out.append(src[i:j + 3])
            i = j + 3
        elif src[i] == '"':
            j = i + 1
            while src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            j = L if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.index("*/", i) + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


_ESC = {"0": "\0", "\\": "\\", "t": "\t", "n": "\n", "r": "\r", '"': '"', "'": "'"}


def _unescape(raw):
    out, i = [], 0
    while i < len(raw):
        c = raw[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        nxt = raw[i + 1]
        if nxt == "u":
            j = raw.index("}", i)
            out.append(chr(int(raw[i + 3:j], 16)))
            i = j + 1
        elif nxt == "(":
            raise ValueError("string interpolation in a tool literal: " + raw[i:i + 40])
        else:
            out.append(_ESC[nxt])
            i += 2
    return "".join(out)


class _Parser:
    """A Swift literal expression: strings (single and multi-line, joined by +), numbers, booleans, and
    array or dictionary literals of those. Anything else raises, so an unparsed tool cannot pass silently."""

    def __init__(self, s, i, names=None):
        self.s, self.i, self.names = s, i, names or {}

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def peek(self, t):
        self.ws()
        return self.s.startswith(t, self.i)

    def eat(self, t):
        if not self.peek(t):
            raise ValueError(f"expected {t!r} at {self.s[self.i:self.i + 60]!r}")
        self.i += len(t)

    def string(self):
        self.ws()
        if self.s.startswith('"""', self.i):
            start = self.s.index("\n", self.i) + 1
            j = start
            while True:
                j = self.s.index('"""', j)
                if self.s[j - 1] != "\\":
                    break
                j += 3
            body = self.s[start:j]
            last_nl = body.rfind("\n")
            indent = body[last_nl + 1:]
            if indent.strip():
                raise ValueError("closing \"\"\" not on its own line")
            lines = body[:last_nl].split("\n") if last_nl >= 0 else []
            out = []
            for k, line in enumerate(lines):
                if line.startswith(indent):
                    line = line[len(indent):]
                elif line.strip():
                    raise ValueError("multi-line string line under-indented")
                else:
                    line = ""
                cont = line.endswith("\\") and not line.endswith("\\\\")
                out.append(line[:-1] if cont else line)
                out.append("" if cont or k == len(lines) - 1 else "\n")
            self.i = j + 3
            return _unescape("".join(out))
        self.eat('"')
        j = self.i
        while self.s[j] != '"':
            j += 2 if self.s[j] == "\\" else 1
        raw = self.s[self.i:j]
        self.i = j + 1
        return _unescape(raw)

    def value(self):
        v = self.term()
        while isinstance(v, str) and self.peek("+"):
            self.eat("+")
            v += self.term()
        return v

    def term(self):
        self.ws()
        c = self.s[self.i]
        if c == '"':
            return self.string()
        m = re.compile(r"[A-Za-z_]\w*").match(self.s, self.i)
        if m and m.group(0) in self.names:
            self.i = m.end()
            return self.names[m.group(0)]
        if c == "[":
            self.eat("[")
            if self.peek(":"):
                self.eat(":")
                self.eat("]")
                return {}
            if self.peek("]"):
                self.eat("]")
                return []
            first = self.value()
            if self.peek(":"):
                d = {}
                self.eat(":")
                d[first] = self.value()
                while self.peek(","):
                    self.eat(",")
                    if self.peek("]"):
                        break
                    k = self.value()
                    self.eat(":")
                    d[k] = self.value()
                self.eat("]")
                return d
            arr = [first]
            while self.peek(","):
                self.eat(",")
                if self.peek("]"):
                    break
                arr.append(self.value())
            self.eat("]")
            return arr
        m = re.compile(r"-?\d+(\.\d+)?").match(self.s, self.i)
        if m:
            self.i = m.end()
            return float(m.group(0)) if m.group(1) else int(m.group(0))
        for word, v in (("true", True), ("false", False)):
            if self.s.startswith(word, self.i):
                self.i += len(word)
                return v
        raise ValueError(f"not a literal: {self.s[self.i:self.i + 60]!r}")


def _tool_at(src, i, names=None):
    """The ClaudeTool(name:description:inputSchema:) literal starting at src[i] ('ClaudeTool('). `names` are
    the function's own `let` constants (FilesystemTool.swift:31, ShellTool.swift:15)."""
    p = _Parser(src, i + len("ClaudeTool("), names)
    out = {}
    for label, key in (("name", "name"), ("description", "description"), ("inputSchema", "schema")):
        p.eat(label)
        p.eat(":")
        out[key] = p.value()
        if label != "inputSchema":
            p.eat(",")
    if p.peek(","):
        p.eat(",")
    p.eat(")")
    return out, p.i


def _body(src, header_re):
    """The text between the braces of the first function whose header matches."""
    m = re.search(header_re, src)
    if not m:
        raise ValueError("no " + header_re)
    i = src.index("{", m.end())
    depth, j = 0, i
    while True:
        if src.startswith('"""', j):
            j = src.index('"""', j + 3) + 3
            continue
        if src[j] == '"':
            j += 1
            while src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j += 1
            continue
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
        j += 1


def _read(path):
    return _strip_comments(open(os.path.join(suite.SRC, path), encoding="utf-8").read())


def _group_file(type_name):
    for root, _, files in os.walk(suite.SRC):
        for f in files:
            if f.endswith(".swift"):
                p = os.path.join(root, f)
                if re.search(rf"\b(enum|struct|class|actor) {type_name}\b", open(p, encoding="utf-8").read()):
                    return os.path.relpath(p, suite.SRC)
    raise ValueError("no declaration of " + type_name)


def _lets(body):
    """Literal `let NAME = ...` constants in a function body, by name."""
    names = {}
    for m in re.finditer(r"\blet ([A-Za-z_]\w*) = ", body):
        try:
            names[m.group(1)] = _Parser(body, m.end(), names).value()
        except (ValueError, IndexError):
            pass
    return names


def _tools_in(body):
    out, i, names = [], 0, _lets(body)
    for m in re.finditer(r"\bClaudeTool\(", body):
        if m.start() < i:
            continue
        t, i = _tool_at(body, m.start(), names)
        out.append(t)
    return out


@functools.lru_cache(maxsize=None)
def _catalogue():
    """ChatService.allTools() (ChatService.swift:1079-1555) in its own order: the inline tools and each
    appended group's claudeTools() literal. MCPManager.shared.claudeTools() is live state, empty until a
    server is set up (ToolCatalogueTests' header), so a fresh install, and this, add none."""
    body = _body(_read("ChatService.swift"), r"static func allTools\(\) -> \[ClaudeTool\]")
    tools, groups, i = [], [], 0
    pat = re.compile(r"\bClaudeTool\(|append\(contentsOf:\s*([A-Za-z_][\w.]*)\.claudeTools\(\)\)")
    while True:
        m = pat.search(body, i)
        if not m:
            break
        if m.group(0).startswith("ClaudeTool("):
            t, i = _tool_at(body, m.start())
            tools.append(t)
            continue
        i = m.end()
        name = m.group(1)
        if name == "MCPManager.shared":
            groups.append((name, None, 0))
            continue
        path = _group_file(name)
        src = _read(path)
        decl = re.search(rf"\b(enum|struct|class|actor) {name}\b", src)
        found = _tools_in(_body(src[decl.start():], r"func claudeTools\(\) -> \[ClaudeTool\]"))
        groups.append((name, path, len(found)))
        tools += found
    return tools, groups


def all_tools():
    return _catalogue()[0]


# --------------------------------------------------------------------------- ToolRelevance (ToolRelevance.swift)

GROUP_PREFIXES = [  # ToolRelevance.group(for:), ToolRelevance.swift:22-34
    ("ios_", "developer"), ("agent_swarm", "developer"), ("shell_", "developer"), ("design_", "developer"),
    ("foundry_", "developer"), ("start_workflow", "developer"), ("save_skill", "developer"), ("backup_", "developer"),
    ("compare_", "developer"),
    ("start_meeting", "meetings"), ("end_meeting", "meetings"), ("enroll_speaker", "meetings"), ("rename_speaker", "meetings"),
    ("rename_cluster", "meetings"), ("list_meetings", "meetings"), ("search_meetings", "meetings"), ("get_meeting", "meetings"),
    ("summarize_meeting", "meetings"), ("move_meeting", "meetings"), ("classify_meeting", "meetings"), ("export_audio", "meetings"),
    ("create_folder", "meetings"), ("rename_folder", "meetings"), ("delete_folder", "meetings"), ("list_folders", "meetings"),
    ("research_web", "research"), ("deep_research", "research"), ("youtube_transcript", "research"), ("list_research", "research"),
    ("slack_", "export"), ("notion_", "export"), ("export_memory", "export"), ("import_memory", "export"),
    ("creative_", "creative"),
    ("read_screen", "screen"), ("control_screen", "screen"), ("run_focus_check", "screen"),
]
KEYWORDS = {  # ToolRelevance.keywords, ToolRelevance.swift:39-55
    "developer": ["build", "builds", "code", "coding", "repo", "repository", "app", "apps", "ios", "iphone", "xcode", "swift",
                  "simulator", "scaffold", "ship", "testflight", "deploy", "publish", "release", "swarm", "agent", "agents",
                  "worker", "workers", "shell", "terminal", "script", "compile", "compiles", "bug", "bugs", "error", "errors",
                  "test", "tests", "git", "github", "pull request", "commit", "design", "mockup", "mock up", "prototype",
                  "landing page", "website", "site", "html", "css", "brand kit", "logo", "skill", "skills", "workflow",
                  "workflows", "backup", "upgrade", "foundry", "compare", "project", "projects", "localize", "run it"],
    "meetings": ["meeting", "meetings", "capture", "transcript", "transcripts", "speaker", "speakers", "enroll", "recording",
                 "record", "standup", "who said", "folder", "folders", "diarize", "export the audio"],
    "research": ["research", "deep dive", "look into", "investigate", "report", "reports", "youtube", "video", "article",
                 "paper", "study", "sources", "summarize this", "what does the internet"],
    "export": ["slack", "notion", "push", "sync", "export", "import", "channel", "workday log", "post to"],
    "creative": ["image", "images", "picture", "pictures", "photo", "render", "generate", "art", "thumbnail", "poster",
                 "illustration", "hero shot", "visual", "visuals", "creative", "bundle"],
    "screen": ["screen", "look at", "what's on", "read that", "click", "type", "scroll", "press", "control", "see this",
               "seeing", "cursor", "button", "form", "fill", "am i focused", "focus check", "what app"],
}
ALL_GROUPS = {"core", *KEYWORDS}


def group(name):
    n = name.lower()
    return next((g for p, g in GROUP_PREFIXES if n.startswith(p)), "core")


def contains_any_word(text, keywords):
    """ChatIntentClassifier.containsAnyWord, IntentClassifier.swift:55-66. NSRegularExpression anchors ^ and
    $ at the ends of the input only, so $ is \\Z here."""
    low = text.lower()
    return any(re.search(r"(?:^|[^a-z])" + re.escape(k) + r"(?:[^a-z]|\Z)", low) for k in keywords)


def groups_for(utterance, recent_tool_names=(), has_image=False):
    """ToolRelevance.groups(for:), ToolRelevance.swift:59-67. An eval turn has no recent tools and no image."""
    text = utterance.strip()
    if not text:
        return set(ALL_GROUPS)
    out = {"core"}
    if has_image:
        out.add("screen")
    out |= {g for g, words in KEYWORDS.items() if contains_any_word(text, words)}
    out |= {group(n) for n in recent_tool_names}
    return out


def relevance_filter(tools, utterance):
    """ToolRelevance.filter, ToolRelevance.swift:69-72."""
    active = groups_for(utterance)
    return [t for t in tools if group(t["name"]) in active]


# --------------------------------------------------------------------------- LocalTurn (LocalTurn.swift)

def _is_letter(ch):
    return unicodedata.category(ch).startswith("L")


def _is_number(ch):
    return unicodedata.category(ch).startswith("N") or unicodedata.numeric(ch, None) is not None


def rank_words(utterance):
    """LocalTurn.swift:89-90: lowercased, split on anything not a letter or number, 4 or more characters."""
    words, cur = set(), []
    for ch in utterance.lower():
        if _is_letter(ch) or _is_number(ch):
            cur.append(ch)
        else:
            if cur:
                words.add("".join(cur))
            cur = []
    if cur:
        words.add("".join(cur))
    return {w for w in words if swift_count(w) >= 4}


def rank(tools, utterance):
    """LocalTurn.rank, LocalTurn.swift:88-103: words found in name-with-spaces plus description, then the
    everyday order, then the original order."""
    words = rank_words(utterance)

    def key(pair):
        off, t = pair
        text = (t["name"].replace("_", " ") + " " + t["description"]).lower()
        score = sum(1 for w in words if w in text)
        ev = EVERYDAY.index(t["name"]) if t["name"] in EVERYDAY else sys.maxsize
        return (-score, ev, off)
    return [t for _, t in sorted(enumerate(tools), key=key)]


def fit(system_chars, message_chars, tool_chars_list, window, reply=REPLY_TOKENS):
    """LocalTurn.fit, LocalTurn.swift:109-138: (messages, tools) that fit."""
    budget = window - reply - MARGIN - tokens(system_chars)
    messages = tools = 0
    if message_chars:
        budget -= tokens(message_chars[-1])
        messages = 1

    def take(limit):
        nonlocal budget, tools
        spent = 0
        while tools < len(tool_chars_list):
            cost = tokens(tool_chars_list[tools])
            if not (cost <= budget and spent + cost <= limit):
                return
            budget -= cost
            spent += cost
            tools += 1
    take(max(0, int(budget / 2)))       # Swift Int division truncates toward zero
    for chars in reversed(message_chars[:-1]):
        cost = tokens(chars)
        if cost > budget:
            break
        budget -= cost
        messages += 1
    take(sys.maxsize)
    return messages, tools


def cut_tools_line(names):
    """LocalTurn.cutToolsLine, LocalTurn.swift:43-48."""
    if not names:
        return None
    return (f"TOOLS NOT OFFERED IN THIS LOCAL TURN: {', '.join(names)}. "
            "If the request needs one of these, say plainly that it cannot be done in this local turn, "
            "and that a bigger local model or a cloud key in Settings, Models enables it.")


def offer(utterance, system_text, volatile_text, memory_gb=16):
    """(offered tools in order, the cut line or None) for one single-message turn, as
    ChatService.assemblePendingContext builds it (ChatService.swift:741-766)."""
    tools = rank(relevance_filter(all_tools(), utterance), utterance)
    worst = cut_tools_line([t["name"] for t in tools])
    system_chars = swift_count(system_text) + swift_count(volatile_text)
    n_msgs, n_tools = fit(system_chars + (swift_count(worst) if worst else 0),
                          [message_chars({"role": "user", "content": utterance})],
                          [tool_chars(t) for t in tools], fit_window(memory_gb))
    return tools[:n_tools], cut_tools_line([t["name"] for t in tools[n_tools:]])


def wire(t):
    """OpenAICompatBackend.openAITools, OpenAICompatBackend.swift:475-486; Ollama takes it as is."""
    return {"type": "function", "function": {"name": t["name"], "description": t["description"],
                                             "parameters": t["schema"]}}


def chat_request(text, model, now_line=None):
    """A realistic local chat turn for one user message: suite.request's chat body with the 3.1.0 tool
    offering, the cut line and the lean reply budget. now_line replaces the NOW line (train_bank)."""
    system = suite.chat_system()
    vol = suite.VOLATILE if now_line is None else suite.VOLATILE.replace("NOW: " + suite.NOW, "NOW: " + now_line, 1)
    vol = vol.strip()
    offered, cut = offer(text, system, vol)
    head = suite.TURN_LABEL + "\n" + vol + ("\n\n" + cut if cut else "")
    user = f"{head}\n\nTHE USER'S MESSAGE:\n{text}"
    req = suite._body(model, system, user, REPLY_TOKENS, 0.5, [wire(t) for t in offered])
    req["offered"] = [t["name"] for t in offered]
    return req


CHAT_FAMILIES = ("chat_tools", "chat_plain", "chat_honesty", "injection")


def chat_text(c):
    """The user's own words in a chat-shaped case (S1, held-out and the training bank)."""
    return c.get("utterance") if c.get("utterance") is not None else c["text"]


# --------------------------------------------------------------------------- self-test

def _self_test():
    errors = []
    tools, groups = _catalogue()
    names = [t["name"] for t in tools]
    print(f"catalogue: {len(tools)} tools from ChatService.allTools() plus {len(groups)} groups")
    if len(tools) != 116:   # ToolCatalogueTests.expectedToolCount at v3.1.0
        errors.append(f"{len(tools)} tools, ToolCatalogueTests pins 116")
    if len(set(names)) != len(names):
        errors.append("duplicate tool names")
    by = {t["name"]: t for t in tools}
    for t in suite.TOOLS:
        f = t["function"]
        got = by.get(f["name"])
        if got is None:
            errors.append(f"suite tool {f['name']} not in the catalogue")
        elif wire(got) != t:
            errors.append(f"suite tool {f['name']} differs from the source")
    # LeanLocalTurnTests.test_theToolsAskedAboutComeFirst (FirstRunWalkW2dTests.swift:202-207)
    toy = [{"name": n, "description": d, "schema": {}} for n, d in
           [("play_music_track", "Play a song"), ("create_note", "Write a note"), ("list_tasks", "List the tasks"),
            ("search_web", "Search the web")]]
    if rank(toy, "write a note about lunch")[0]["name"] != "create_note":
        errors.append("rank: the note tool does not lead 'write a note about lunch'")
    if rank(toy, "hi")[0]["name"] != "list_tasks":
        errors.append("rank: the everyday tools do not lead an unclear message")
    # test_fitKeepsTheNewestMessageAndPrefersToolsToOldHistory (FirstRunWalkW2dTests.swift:209-215)
    m, n = fit(7_000, [2_000] * 20 + [100], [700] * 50, 8_192)
    if not (m >= 1 and n > 0 and m < 21):
        errors.append(f"fit toy case gave {m} messages, {n} tools")
    # cut line: names exactly the tools left out, none when nothing is
    if cut_tools_line([]) is not None:
        errors.append("cut line on nothing")
    # a fit with room offers every relevant tool and names none (LeanTurnHonestyTests' control)
    allr = rank(relevance_filter(tools, "what is on my plate today?"), "what is on my plate today?")
    if fit(0, [10], [tool_chars(t) for t in allr], 10 ** 7)[1] != len(allr):
        errors.append("fit with room drops tools")
    # the window does not depend on 8 GB against 16 GB
    if fit_window(8) != fit_window(16) or fit_window(16) != 8_192:
        errors.append("fit window by memory")
    return errors


def _table():
    sys.path.insert(0, os.path.join(HERE, "s2"))
    import holdout
    rows = [c for c in holdout.cases() if c["family"] in CHAT_FAMILIES]
    sizes, missing, cut_n = [], [], 0
    for c in rows:
        r = chat_request(chat_text(c), "m")
        sizes.append(len(r["offered"]))
        if c["family"] == "chat_tools" and c["tool"] not in r["offered"]:
            missing.append((c["id"], c["tool"], c["utterance"]))
        cut_n += "TOOLS NOT OFFERED" in r["body"]["messages"][1]["content"]
    print(f"held-out chat turns: {len(rows)}; tools offered min {min(sizes)}, median {sorted(sizes)[len(sizes) // 2]}, "
          f"max {max(sizes)}; {cut_n} turns carry the cut line")
    print(f"chat_tools cases whose expected tool is not offered: {len(missing)}")
    for m in missing:
        print("  ", m)


if __name__ == "__main__":
    errs = _self_test()
    for e in errs:
        print("ERROR", e)
    print("SELF-TEST", "FAIL" if errs else "PASS", f"({len(errs)} errors)")
    if not errs:
        _table()
    sys.exit(1 if errs else 0)
