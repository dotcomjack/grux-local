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


def _unescape(raw, names=None):
    """A Swift string literal's body. `\\(NAME)` is filled in only when NAME is a literal constant in `names`
    (ShellRecoveryTools.swift interpolates its own `static let` numbers into descriptions); any other
    interpolation raises."""
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
            j = raw.find(")", i)
            key = raw[i + 2:j] if j > 0 else ""
            if not names or not isinstance(names.get(key), (int, str)) or isinstance(names.get(key), bool):
                raise ValueError("string interpolation in a tool literal: " + raw[i:i + 40])
            out.append(str(names[key]))
            i = j + 1
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
            return _unescape("".join(out), self.names)
        self.eat('"')
        j = self.i
        while self.s[j] != '"':
            j += 2 if self.s[j] == "\\" else 1
        raw = self.s[self.i:j]
        self.i = j + 1
        return _unescape(raw, self.names)

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
        m = re.compile(r"\[\s*String\s*:\s*Any\s*\]\(\)").match(self.s, self.i)
        if m:   # an empty dictionary written as a typed initializer (ShellRecoveryTools.swift)
            self.i = m.end()
            return {}
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


def _read(path, src=None):
    return _strip_comments(open(os.path.join(src or suite.SRC, path), encoding="utf-8").read())


def _group_file(type_name, src=None):
    src = src or suite.SRC
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith(".swift"):
                p = os.path.join(root, f)
                if re.search(rf"\b(enum|struct|class|actor) {type_name}\b", open(p, encoding="utf-8").read()):
                    return os.path.relpath(p, src)
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


def _tools_in(body, outer=None):
    """Every ClaudeTool literal in a function body. `outer` are the enclosing type's own literal constants
    (ShellRecoveryTools.swift names its tools with `static let`); the function's lets win."""
    out, i, names = [], 0, {**(outer or {}), **_lets(body)}
    for m in re.finditer(r"\bClaudeTool\(", body):
        if m.start() < i:
            continue
        t, i = _tool_at(body, m.start(), names)
        out.append(t)
    return out


@functools.lru_cache(maxsize=None)
def _catalogue(src=None):
    """ChatService.allTools() (ChatService.swift:1079-1555) in its own order: the inline tools and each
    appended group's claudeTools() literal. MCPManager.shared.claudeTools() is live state, empty until a
    server is set up (ToolCatalogueTests' header), so a fresh install, and this, add none."""
    body = _body(_read("ChatService.swift", src), r"static func allTools\(\) -> \[ClaudeTool\]")
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
        path = _group_file(name, src)
        text = _read(path, src)
        decl = re.search(rf"\b(enum|struct|class|actor) {name}\b", text)
        found = _tools_in(_body(text[decl.start():], r"func claudeTools\(\) -> \[ClaudeTool\]"),
                          _lets(_body(text[decl.start():], rf"\b(?:enum|struct|class|actor) {name}\b")))
        groups.append((name, path, len(found)))
        tools += found
    return tools, groups


def all_tools(src=None):
    """ChatService.allTools() read from `src` (a Grux-Mac/Sources/Grux tree), suite.SRC when not given."""
    return _catalogue(src)[0]


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


# =========================================================================== Grux 3.3.1
#
# The local turn of Grux tag v3.3.1 (commit 4b8d1b6e), the current release, ported beside the 3.1.0 turn above,
# which stays byte for byte as it was (held-out v2 is frozen on it). Lines below cite v3.3.1. Read from
# GRUX_SRC_V3 (recipe/get_grux_source.sh with GRUX_TAG=v3.3.1), never from suite.SRC.
#
# Every 3.3.1 change that reaches the request, against the 3.1.0 port:
#   tools      catalogue 119 (ShellRecoveryTools.claudeTools, ChatService.swift:2023; remember_slang's description,
#              ChatService.swift:1804); a new ToolRelevance group, otherApps (open_url, open_app, play_on_youtube,
#              search_web), switched on by its own words or a web address (ToolRelevance.swift:41, 64-75, 109-111);
#              "audit log" switches on developer (ToolRelevance.swift:54); the starter-chip shapes switch on
#              developer and lead with their recovery tools (ToolRelevance.swift:78-87, 108; ChatService.swift:1137-1141);
#              rank by whole words with stems, no common words, to-do read as task (LocalTurn.swift:134-158);
#              fit: the tools' half is taken before the newest message, capped by what is left after it, and there
#              is no second pass (LocalTurn.swift:309-344); the per-turn block and the longest cut line are counted
#              on the newest message, the stable prompt alone on the system side (ChatService.swift:1143-1156);
#              the stable list: one fixed core list most turns take (LocalTurn.swift:170-238, ChatService.swift:
#              1162-1178, 1226-1253); offered tools and the cut line in one canonical order (LocalTurn.swift:
#              361-379, ChatService.swift:1161, 1173, 1180-1184); no tools at all on a draft, list or pasted-text
#              turn (ChatService.swift:1126-1128); a Grux Local tag ending -v4 gets one-line signatures
#              (TrainedTools.swift:121-124, ChatService.swift:1196).
#   system     ABOUT_YOU after the persona (ChatService.swift:2891, 2897; AboutYou.swift:11-34); a new tool note
#              (LocalTurn.swift:80-82).
#   per turn   a weather note and a draft, list or pasted-text note at the end of the volatile block
#              (ChatService.swift:3254-3263, 3278); a prefilled assistant message the model continues
#              (ChatService.swift:1188-1190, OpenAICompatBackend.swift:242-250).
#   body       "think": false on every native call, "keep_alive": "30m" to this Mac's server, keys sorted at
#              every level (OpenAICompatBackend.swift:212-214, 296-309).
#   no request at all: 3.3.1 answers 16 more kinds of ask with no model (ChatService.swift:271-540, 609-623);
#              the Commands V2 triggers, PIM and the confidence gate answer without one in 3.1.0 too.
# Unchanged: num_predict 2,000 (ChatService.swift:792, 1062; the 3.1.0 port above sends 1,024, which is the fit's
# reply budget, not the wire's: held-out v2 is frozen with it), temperature 0.5, num_ctx by memory, PROMPT_CEILING,
# the cut line's words.
#
# The per-utterance classifiers (the fast paths, DraftAsk, ListAsk, WeatherAsk.localNote,
# ChatIntentClassifier.classify and .pimRoute, GreetingAsk, ConfidenceGate) are NOT re-ported: their answers come
# from that Swift compiled as is (recipe/grux_oracle), frozen per utterance in local_turn_v331_asks.json. The
# rules ported below are checked against the same compiled Swift on every held-out turn
# (recipe/test_local_turn_v331.py).

V331_TAG = "v3.3.1"
V331_COMMIT = "4b8d1b6e31f7dd98e3f469816098d6adac9b1f7a"
V331_TOOLS = 119                     # ToolCatalogueTests.expectedToolCount at v3.3.1
KEEP_ALIVE = "30m"                   # LocalTurn.keepAlive, LocalTurn.swift:64
CHAT_MAX_TOKENS = 2_000              # ChatService.chatMaxTokens, ChatService.swift:1062: what streamCompleteWithTools
#                                      sends (ChatService.swift:792). LocalTurn.replyTokens (1,024) only sizes the fit
#                                      and the cost estimate (ChatService.swift:764, 1203); 3.1.0 also sent 2000
#                                      (ChatService.swift:488 at v3.1.0), so held-out v2's 1,024 is not the app's.
TURN_RESERVE_CHARS = 2_700           # LocalTurn.turnReserveChars, LocalTurn.swift:176
POINTS_PAST_SCORE = 3                # LocalTurn.pointsPastScore, LocalTurn.swift:180
GROUP_POINTS_PAST_SCORE = 2          # LocalTurn.groupPointsPastScore, LocalTurn.swift:186
HISTORY_MESSAGES, HISTORY_STEP = 30, 10   # LocalTurn.swift:268-269
ASSISTANT_NAME = "Jax"               # UserIdentity.assistantName on a fresh install, UserIdentity.swift:48-51
ASKS_PATH = os.path.join(HERE, "local_turn_v331_asks.json")
TOOL_NOTE_V331 = ("TOOLS: call a tool only when the user asks for something it does (a task, an event, a note, a "
                  "search). A question about you (what you can do, who you are, how you work) needs no tool: answer "
                  "it from WHAT_YOU_CAN_DO below. Otherwise answer directly, in one to three sentences.")
#                                    LocalTurn.toolNote, LocalTurn.swift:80-82
COMMON_WORDS = {  # LocalTurn.commonWords, LocalTurn.swift:99-108
    "what", "whats", "when", "where", "which", "while", "whom", "whose", "that", "this", "these", "those",
    "there", "their", "theirs", "them", "they", "then", "than", "have", "having", "with", "without", "from",
    "your", "yours", "about", "could", "would", "should", "please", "just", "like", "some", "will", "into",
    "onto", "over", "also", "been", "were", "does", "doing", "done", "make", "made", "know", "tell", "give",
    "need", "want", "thanks", "thank", "okay", "sure", "much", "many", "more", "most", "very", "really",
    "only", "even", "still", "here", "same", "each", "every", "other", "another", "such", "both", "after",
    "before", "again", "being", "dont", "doesn", "didn", "cant", "plus", "minus",
    "times", "divided", "equals", "hello", "thing", "things", "something", "anything", "nothing", "everything",
}
FILLER_WORDS = {  # LocalTurn.fillerWords, LocalTurn.swift:196-203
    "about", "above", "after", "again", "also", "been", "before", "being", "both", "could", "does", "doing",
    "down", "each", "from", "have", "having", "here", "into", "just", "more", "most", "much", "only",
    "other", "over", "please", "same", "should", "some", "such", "than", "that", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "under", "until", "very", "were", "what",
    "when", "where", "which", "while", "with", "would", "your", "yours", "okay", "thanks", "thank",
    "anything", "something",
}
GROUP_PREFIXES_V331 = GROUP_PREFIXES + [  # ToolRelevance.group(for:), ToolRelevance.swift:41
    ("open_url", "otherApps"), ("open_app", "otherApps"), ("play_on_youtube", "otherApps"), ("search_web", "otherApps")]
KEYWORDS_V331 = {g: list(w) for g, w in KEYWORDS.items()}
KEYWORDS_V331["developer"] = KEYWORDS["developer"] + ["audit log"]          # ToolRelevance.swift:53-54
KEYWORDS_V331["otherApps"] = [  # ToolRelevance.swift:64-69
    "launch", "open", "opens", "opening", "pull up", "bring up", "go to", "show me", "fire up", "boot up",
    "start", "switch to", "visit", "navigate", "youtube", "video", "videos", "watch", "play", "playing",
    "put on", "throw on", "queue up", "hear", "listen", "listening", "song", "songs", "music", "album",
    "playlist", "track", "chrome", "safari", "firefox", "browser", "tab", "website", "site", "page",
    "url", "link", "app", "google", "search", "search for", "look up", "online", "web", "http",
    "https", "www"]
WEB_ADDRESS = re.compile(  # ToolRelevance.webAddress, ToolRelevance.swift:73-74
    r"(?:^|[^a-z0-9.-])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|dev|app|ai|co|edu|gov|me|tv|us|uk)(?:[^a-z0-9-]|$)")
CHIP_SHAPES = ["undo `", "why did you refuse `", "what did you run today?"]   # ToolRelevance.swift:77
AUDIT_TOOL, LIST_TOOL, RESTORE_TOOL = "shell_audit_log", "shell_snapshots", "shell_restore_snapshot"
#                                    ShellRecoveryTools.swift:18-20
# The order send() tries the asks answered with no model, ChatService.swift:282-540 then 609-623 (after the
# readiness guard) and 627-664 (the confidence gate, heuristic only: ConfidenceGate.modelPathEnabled is false).
FAST_PATHS = ("upgrade_yourself", "capability", "privacy", "privacy_kind", "voice", "self", "todo_list", "math",
              "day_count", "date", "foreign_note", "time", "calendar", "calendar_search", "weather", "never_mind",
              "pim", "greeting", "clarify")


def src_v331():
    """The Grux-Mac/Sources/Grux tree at v3.3.1. Refuses a missing or wrong tree: its catalogue must be 119."""
    s = os.environ.get("GRUX_SRC_V3")
    if not s:
        raise SystemExit('the 3.3.1 turn reads GRUX_SRC_V3: export GRUX_SRC_V3="$(GRUX_TAG=v3.3.1 sh '
                         'recipe/get_grux_source.sh)"')
    return s


_count = functools.lru_cache(maxsize=65_536)(swift_count)   # String.count, memoized: the same texts come back


def _tool_chars(t):
    """LocalTurn.chars(_ t:), LocalTurn.swift:382-384 (unchanged since 3.1.0: tool_chars), with the counts memoized."""
    return _count(t["name"]) + _count(t["description"]) + json_bytes(t["schema"]) + 40


@functools.lru_cache(maxsize=4_096)
def _tool_words(name, description):
    """The words of a tool's name and description that rank matches against, LocalTurn.swift:142-143."""
    return frozenset(_split_words((name.replace("_", " ") + " " + description).lower()))


@functools.lru_cache(maxsize=None)
def _tools_v331_cached(s):
    tools = all_tools(s)
    if len(tools) != V331_TOOLS or RESTORE_TOOL not in {t["name"] for t in tools}:
        raise SystemExit(f"{s} defines {len(tools)} tools; Grux {V331_TAG} defines {V331_TOOLS}. Point GRUX_SRC_V3 at "
                         f"that tag.")
    return tools


def all_tools_v331():
    """ChatService.allTools() at v3.3.1, ChatService.swift:1732."""
    return _tools_v331_cached(src_v331())


# --------------------------------------------------------------------------- ToolRelevance at 3.3.1

def group_v331(name):
    """ToolRelevance.group(for:), ToolRelevance.swift:27-46."""
    n = name.lower()
    return next((g for p, g in GROUP_PREFIXES_V331 if n.startswith(p)), "core")


def chip_tools(utterance):
    """ToolRelevance.chipTools(for:), ToolRelevance.swift:80-87."""
    low = utterance.strip().lower()
    if low.startswith("undo `"):
        return [LIST_TOOL, RESTORE_TOOL]
    if low.startswith("why did you refuse `") or low.startswith("what did you run today?"):
        return [AUDIT_TOOL]
    return []


def groups_for_v331(utterance):
    """ToolRelevance.groups(for:), ToolRelevance.swift:99-116, for an eval turn: no image, no tool run recently."""
    text = utterance.strip()
    if not text:
        return {"core", *KEYWORDS_V331}
    out = {"core"} | {g for g, words in KEYWORDS_V331.items() if contains_any_word(text, words)}
    low = text.lower()
    if any(low.startswith(s) for s in CHIP_SHAPES):
        out.add("developer")
    if WEB_ADDRESS.search(low):
        out.add("otherApps")
    return out


def relevance_filter_v331(tools, utterance):
    """ToolRelevance.filter, ToolRelevance.swift:118-121."""
    active = groups_for_v331(utterance)
    return [t for t in tools if group_v331(t["name"]) in active]


# --------------------------------------------------------------------------- LocalTurn at 3.3.1

def _split_words(text):
    """`split { !$0.isLetter && !$0.isNumber }` over a lowercased string, kept in order."""
    out, cur = [], []
    for ch in text:
        if _is_letter(ch) or _is_number(ch):
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def rank_v331(tools, utterance):
    """LocalTurn.rank, LocalTurn.swift:134-158 (D17, D180): whole words, no common words, a plural finds its
    singular, "to-do" is "task"; then the everyday order, then the original order."""
    said = re.sub(r"\bto-?dos?\b|\bto do(?= list)", "task", utterance.lower())
    words = {w for w in _split_words(said) if _count(w) >= 4 and w not in COMMON_WORDS}

    def score(t):
        text = _tool_words(t["name"], t["description"])
        n = 0
        for w in words:
            stem = w[:-1] if w.endswith("s") and _count(w) > 4 else w
            n += 1 if (w in text or stem in text or stem + "s" in text) else 0
        return n

    def key(pair):
        off, t = pair
        ev = EVERYDAY.index(t["name"]) if t["name"] in EVERYDAY else sys.maxsize
        return (-score(t), ev, off)
    return [t for _, t in sorted(enumerate(tools), key=key)]


def substring_score(t, words):
    """LocalTurn.score(_:words:), LocalTurn.swift:120-123: the 3.1.0 substring scoring pointsPast still uses."""
    text = (t["name"].replace("_", " ") + " " + t["description"]).lower()
    return sum(1 for w in words if w in text)


STABLE_HEAD = [n for n in EVERYDAY + ["set_mode", "create_note", "remember_fact", "remove_task"]
               if group_v331(n) == "core"]   # LocalTurn.stableHead, LocalTurn.swift:170-171


def order_key(name):
    """LocalTurn.orderKey, LocalTurn.swift:215-223."""
    g = group_v331(name)
    band = 0 if g == "core" else 2 if g == "otherApps" else 1
    return band, (STABLE_HEAD.index(name) if name in STABLE_HEAD else sys.maxsize)


def canonical_order(tools):
    """LocalTurn.canonicalOrder, LocalTurn.swift:361-371: name to place."""
    ranked = sorted(enumerate(tools), key=lambda p: (*order_key(p[1]["name"]), p[0]))
    out = {}
    for i, (_, t) in enumerate(ranked):
        out.setdefault(t["name"], i)
    return out


def in_canonical_order(tools, order):
    """LocalTurn.inCanonicalOrder, LocalTurn.swift:374-379."""
    return [t for _, t in sorted(enumerate(tools), key=lambda p: (order.get(p[1]["name"], sys.maxsize), p[0]))]


def stable_order(tools):
    """LocalTurn.stableOrder, LocalTurn.swift:208-210."""
    return in_canonical_order(tools, canonical_order(tools))


def fit_v331(system_chars, message_chars, tool_chars_list, window, reply=REPLY_TOKENS, tools_take_all_room=False):
    """LocalTurn.fit, LocalTurn.swift:309-344: (messages, tools)."""
    budget = window - reply - MARGIN - tokens(system_chars)
    messages = tools = 0
    tool_limit = max(0, budget) if tools_take_all_room else max(0, int(budget / 2))
    if message_chars:
        budget -= tokens(message_chars[-1])
        tool_limit = min(tool_limit, max(0, budget))
        messages = 1
    spent = 0
    while tools < len(tool_chars_list):
        cost = tokens(tool_chars_list[tools])
        if not (cost <= budget and spent + cost <= tool_limit):
            break
        budget -= cost
        spent += cost
        tools += 1
    for chars in reversed(message_chars[:-1]):
        cost = tokens(chars)
        if cost > budget:
            break
        budget -= cost
        messages += 1
    return messages, tools


def history_start(count, fitting):
    """LocalTurn.historyStart, LocalTurn.swift:278-283."""
    if count <= 0:
        return 0
    over = max(0, count - HISTORY_MESSAGES)
    window = (over + HISTORY_STEP - 1) // HISTORY_STEP * HISTORY_STEP
    return min(max(window, count - max(1, fitting)), count - 1)


def stable_tools(core, cached_system_chars, longest_cut_line_chars, window):
    """LocalTurn.stableTools, LocalTurn.swift:231-238: the core list in the stable order, as many as fit beside the
    system text counted in whole 2,000 character steps, the longest cut line and the turn reserve."""
    ordered = stable_order(core)
    stepped = (cached_system_chars // 2_000 + 1) * 2_000
    _, n = fit_v331(stepped + longest_cut_line_chars + TURN_RESERVE_CHARS, [], [_tool_chars(t) for t in ordered],
                    window, tools_take_all_room=True)
    return ordered[:n]


def points_past(stable, candidates, utterance):
    """LocalTurn.pointsPast, LocalTurn.swift:248-261."""
    words = rank_words(utterance) - FILLER_WORDS
    inside = {t["name"] for t in stable}
    scores = [(t["name"], substring_score(t, words)) for t in stable]
    best = max((s for _, s in scores), default=0)
    everyday_leads = best > 0 and any(s == best and n in EVERYDAY for n, s in scores)
    for t in candidates:
        if t["name"] in inside:
            continue
        s = substring_score(t, words)
        if s >= POINTS_PAST_SCORE and s > best:
            return True
        if group_v331(t["name"]) == "core" or s < GROUP_POINTS_PAST_SCORE:
            continue
        if s > best or (s == best and not everyday_leads):
            return True
    return False


def _block_chars(blocks):
    """`chars(_ blocks:)`, ChatService.swift:1145-1147: the blocks' text, String.count."""
    return sum(_count(b["text"]) for b in blocks)


def stable_local_tools(system_blocks, messages, ranked, utterance, sizing_window, context_window, why=None):
    """ChatService.stableLocalTools, ChatService.swift:1226-1253, for a turn with no image, no preset and no tool
    run in the last fifteen minutes. None when the turn takes the ranked list; `why` (a list) gets the reason."""
    def no(reason):
        if why is not None:
            why.append(reason)
        return None
    if any(group_v331(t["name"]) == "otherApps" for t in ranked):
        return no("other_apps")            # ChatService.swift:1232 (D12)
    every = all_tools_v331()
    stable = stable_tools([t for t in every if group_v331(t["name"]) == "core"], _block_chars(system_blocks[:1]),
                          _count(cut_tools_line([t["name"] for t in every])), sizing_window)
    names = {t["name"] for t in stable}
    if not stable:
        return no("empty")
    if points_past(stable, ranked, utterance):
        return no("points_past")           # ChatService.swift:1245-1246
    cut = cut_tools_line([t["name"] for t in ranked if t["name"] not in names])
    prompt = tokens(_block_chars(system_blocks) + (_count(cut) if cut else 0)) \
        + sum(tokens(message_chars(m)) for m in messages) + sum(tokens(_tool_chars(t)) for t in stable)
    if prompt + REPLY_TOKENS + MARGIN > context_window:
        return no("does_not_fit")          # ChatService.swift:1251
    return stable


# --------------------------------------------------------------------------- TrainedTools (Grux Local -v4 tags)

TRAINED_SUFFIX = "-v4"                    # TrainedTools.tagSuffix, TrainedTools.swift:27
ABBREVIATIONS = ["e.g.", "i.e.", "etc.", "vs.", "approx."]   # TrainedTools.swift:63


def is_trained_model(model_id):
    """TrainedTools.isTrainedModel, TrainedTools.swift:33-38."""
    mid = model_id.strip().lower()
    if ":" not in mid:
        return False
    colon = mid.rindex(":")
    repo = mid[:colon].split("/")[-1] if mid[:colon] else ""
    return repo == "grux-local" and mid[colon + 1:].endswith(TRAINED_SUFFIX)


def sorted_wire_bytes(obj):
    """TrainedTools.wire / OpenAICompatBackend.wireData, TrainedTools.swift:48-50: JSONSerialization with sorted keys."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).replace("/", "\\/").encode()


def schema_hash(t):
    """TrainedTools.schemaHash, TrainedTools.swift:54-56."""
    import hashlib
    return hashlib.sha256(sorted_wire_bytes({"name": t["name"], "description": t["description"],
                                             "parameters": t["schema"]})).hexdigest()


@functools.lru_cache(maxsize=None)
def trained_v4(s):
    """TrainedTools.v4, TrainedToolsV4.swift: tool name to the schema hash Grux Local v4 is trained on."""
    text = open(os.path.join(s, "Chat", "TrainedToolsV4.swift"), encoding="utf-8").read()
    return dict(re.findall(r'"([a-z_0-9]+)": "([0-9a-f]{64})"', text))


def first_sentence(text, limit=160):
    """TrainedTools.firstSentence, TrainedTools.swift:66-82."""
    flat = " ".join(text.split())
    end = len(flat)
    for i, c in enumerate(flat):
        if c in ".!?" and (i + 1 == len(flat) or flat[i + 1] == " "):
            if not any(flat[:i + 1].lower().endswith(a) for a in ABBREVIATIONS):
                end = i + 1
                break
    sentence = flat[:end]
    if _count(sentence) <= limit:
        return sentence
    cut = sentence[:limit]
    return (cut[:cut.rindex(" ")] if " " in cut else cut) + "..."


def _type_name(p):
    """TrainedTools.typeName, TrainedTools.swift:85-90."""
    if isinstance(p.get("enum"), list):
        return "|".join(str(o) for o in p["enum"])
    kind = p.get("type") if isinstance(p.get("type"), str) else "any"
    if kind == "array" and isinstance(p.get("items"), dict):
        return f"[{_type_name(p['items'])}]"
    return kind


def signature(t):
    """TrainedTools.signature, TrainedTools.swift:95-104."""
    props = t["schema"].get("properties") or {}
    required = [r for r in (t["schema"].get("required") or []) if r in props]
    optional = sorted(k for k in props if k not in required)
    params = [f"{n}: {_type_name(props[n] or {})}" for n in required] + \
             [f"{n}?: {_type_name(props[n] or {})}" for n in optional]
    return f"{t['name']}({', '.join(params)}): {first_sentence(t['description'])}"


def for_model(model_id, tools):
    """TrainedTools.forModel, TrainedTools.swift:121-124: a trained, unchanged tool as its signature on a -v4 tag."""
    if not is_trained_model(model_id):
        return tools
    v4 = trained_v4(src_v331())
    return [dict(t, description=signature(t)) if v4.get(t["name"]) == schema_hash(t) else t for t in tools]


# --------------------------------------------------------------------------- the system and per-turn text at 3.3.1

def _literal_after(text, marker, names):
    """The Swift string literal opening at the first triple quote after `marker`, `\\(NAME)` filled from names."""
    i = text.index(marker)
    return _Parser(text, text.index('"""', i), names).string()


def display_name(model_id):
    """ComposerFooter.displayName(id:registryName: nil), ComposerFooter.swift:39-62."""
    s = model_id
    if "/" in s:
        s = s[s.rindex("/") + 1:]
    if ":" in s:
        s = s[:s.index(":")]
    if s.startswith("claude-"):
        s = s[len("claude-"):]
    s = re.sub(r"-\d{8}$", "", s)
    if not s:
        return model_id
    words, joinable = [], False
    for token in [x for x in s.split("-") if x]:
        short = len(token) <= 2 and token.isascii() and token.isdigit()
        if short and joinable and words:
            words.append(words.pop() + "." + token)
        else:
            words.append(token[:1].upper() + token[1:])
        joinable = short
    return " ".join(words) if words else model_id


def about_you(model_id):
    """AboutYou.block, AboutYou.swift:11-34, for a fresh install whose chat runs on `model_id` through this Mac's
    Ollama: route .thisMac (PrivacyAsk.swift:217-231, LocalTurn.isLoopback), offline mode off (Models.swift:944),
    Local Models behind the developer door with the developer surfaces locked (FeatureDisposition.swift:95,
    Models.swift:1028, OnboardingSteps.swift:1133-1144, OrbCommandPalette.swift:347-349)."""
    return "\n".join([
        "ABOUT_YOU (true facts about you and this app; answer any question about yourself from these, never guess):",
        f"- Your name is {ASSISTANT_NAME}. The app is Grux AI. It is free and needs no account.",
        f"- Your replies come from {display_name(model_id)}, a model running on this Mac, so chat works with no "
        "internet connection.",
        "- A bigger or smarter model: Local Models has the full list and can download any of them. Open it from the "
        "command palette, Command-Shift-P. Or add a cloud API key in Settings, Models.",   # SettingsSearchRegistry.swift:21
        "- Reading replies aloud: “Speak Grux's replies aloud” in Tuning, under Talks back. "   # VoiceAsk.swift:123, TuningView.swift:92
        "To quiet you at once: the speaker button at the top of Chat, beside Clear.",                     # VoiceAsk.swift:131
    ])


def _persona_v331(s):
    """JaxProfile.persona for a person nothing is learned about, JaxProfile.swift:170-179 (as holdout_v3 renders it)."""
    jax = _read("Jax/JaxProfile.swift", s)
    return (_literal_after(jax, "private static func newHeader(name: String) -> String {", {"name": ASSISTANT_NAME})
            + "\n\n" + _literal_after(jax, "private static func guardrails(name: String) -> String {",
                                      {"name": ASSISTANT_NAME}))


@functools.lru_cache(maxsize=None)
def _chat_source_v331(s):
    """ChatService.swift at v3.3.1, comments blanked, the compound interpolations of its literals named."""
    chat = _read("ChatService.swift", s).replace("\\(KnownProjects.asSystemContext())", "\\(knownProjects)")
    for a, b in _VOLATILE_SUBS.items():
        chat = chat.replace(a, b)
    return chat


@functools.lru_cache(maxsize=None)
def lean_system_v331(model_id):
    """The cached block of a lean 3.3.1 turn: the stable literal (ChatService.swift:2892-3029) with this install's
    persona, no user name (UserIdentity.systemPromptLine, an empty line), ABOUT_YOU, cut at the tool manual, then
    the tool note and the tail (LocalTurn.leanStable, LocalTurn.swift:86-91; ChatService.swift:3288). The tail is
    JAX_IDENTITY's header alone (nothing learned, JaxProfile.swift:215-248) and an empty profile; WHAT_YOU_CAN_DO is
    left out, as held-out v2 leaves it out (it depends on what the install set up)."""
    s = src_v331()
    chat = _chat_source_v331(s)
    empty = {k: "" for k in ("identityLine", "knownProjects", "jaxIdentity", "profile", "skillsContext", "capabilities")}
    stable = _literal_after(chat, 'let stable = """\n        \\(jaxPersona)',
                            {"jaxPersona": _persona_v331(s), "aboutYou": about_you(model_id), **empty})
    head = stable[:stable.index("TOOL USE - READ CAREFULLY")] if "TOOL USE - READ CAREFULLY" in stable else stable
    parts = [head.strip(), TOOL_NOTE_V331,
             "JAX_IDENTITY (how the user reasons, treat this as your own operating mind):", "", ""]
    return "\n\n".join(p.strip() for p in parts if p.strip())


_VOLATILE_SUBS = {  # each interpolation in the volatile literals that is not a plain name, ChatService.swift:3219-3279
    '\\(tasks.isEmpty ? "(empty)" : tasks)': "\\(TASKS)",
    '\\(actions.isEmpty ? "(none)" : actions)': "\\(ACTIONS)",
    '\\(recent.isEmpty ? "(nothing recent)" : recent)': "\\(RECENT)",
    "\\(VoiceMacroRegistry.shared.systemPromptBlock())": "\\(MACROS)",
    '\\(recentMemories.isEmpty ? "(none)" : recentMemories)': "\\(RECENT_MEMORIES)",
    '\\(memoryBlock ?? "")': "\\(MEMORY_BLOCK)",
}


def volatile_v331(asks):
    """The uncached per-turn block of a lean 3.3.1 turn, read from its literals (ChatService.swift:3219-3279) for a
    fresh install: no tasks, actions, focus events or macros (VoiceMacros.swift:579-581), no ambient or pending
    memories, nothing retrieved, nobody known named; the NOW, mode, music, task and app lines are suite.VOLATILE's
    (unchanged since 3.1.0). Which sections ride is ChatIntentClassifier.classify's call (asks["sections"]); the
    weather note and the draft, list or pasted-text note are the app's own (asks["notes"])."""
    chat = _chat_source_v331(src_v331())
    lines = dict(line.split(": ", 1) for line in suite.VOLATILE.strip().split("\n")[:5])
    task, proposed, focus, macros = asks["sections"]
    blocks = {
        "taskStackBlock": _literal_after(chat, "let taskStackBlock = intent.taskStack ? ", {"TASKS": "(empty)"}) if task else "",
        "proposedActionsBlock": _literal_after(chat, "let proposedActionsBlock = intent.proposedActions ? ",
                                               {"ACTIONS": "(none)"}) if proposed else "",
        "recentFocusBlock": _literal_after(chat, "let recentFocusBlock = intent.recentFocus ? ",
                                           {"RECENT": "(nothing recent)"}) if focus else "",
        "macrosBlock": _literal_after(chat, "let macrosBlock = intent.availableMacros ? ",
                                      {"MACROS": "(no macros registered)"}) if macros else "",
    }
    notes = asks["notes"]
    draft = notes["draft"] or notes["list"] or notes["pasted"]
    names = {"nowStr": lines["NOW"], "mode.label": lines["CURRENT_MODE"].split(" - ", 1)[0],
             "mode.replyCapDescription": lines["CURRENT_MODE"].split(" - ", 1)[1],
             "state.config.musicStrategy.promptDirective": lines["MUSIC_STRATEGY"], "current": lines["CURRENT_TASK"],
             "activeAppLine": "ACTIVE_APP_RIGHT_NOW: " + lines["ACTIVE_APP_RIGHT_NOW"], "personContextBlock": "",
             "RECENT_MEMORIES": "(none)", "pendingMemoriesBlock": "(inbox clear - nothing waiting)",
             "MEMORY_BLOCK": "", "weatherBlock": "\n\n" + notes["weather"] if notes["weather"] else "",
             "draftBlock": "\n\n" + draft if draft else "", **blocks}
    return _literal_after(chat, "let volatile = ", names)


def system_blocks_v331(model_id, asks):
    """ChatService.composeSystemBlocks(stable:developer: nil, summary: nil, volatile:), ChatService.swift:3301-3316."""
    return [{"type": "text", "text": lean_system_v331(model_id), "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": volatile_v331(asks)}]


# --------------------------------------------------------------------------- the asks the app answers itself

_ASKS = None


def asks_v331(text):
    """What Grux 3.3.1's own classifiers say about one message (recipe/grux_oracle, frozen in ASKS_PATH): the fast
    path that answers it with no model, if any; whether it is a draft, list or pasted-text turn (no tools, a note, a
    prefilled reply start); the weather note; ChatIntentClassifier.classify's four sections. A message not in the
    table is classified by the oracle when swiftc and GRUX_SRC_V3 are there, else this raises: never a guess."""
    global _ASKS
    if _ASKS is None:
        _ASKS = json.load(open(ASKS_PATH, encoding="utf-8"))["asks"] if os.path.exists(ASKS_PATH) else {}
    if text not in _ASKS:
        sys.path.insert(0, os.path.join(HERE, "grux_oracle"))
        import oracle
        _ASKS.update(oracle.asks([text]))
    return _ASKS[text]


@functools.lru_cache(maxsize=None)
def commands_v2_triggers(s):
    """Every built-in Commands V2 voice trigger, read from the definitions' literals (CommandsV2/*.swift). A fresh
    install has no user-defined command."""
    out = []
    for f in sorted(os.listdir(os.path.join(s, "CommandsV2"))):
        if f.endswith(".swift"):
            src = _read(os.path.join("CommandsV2", f), s)
            for m in re.finditer(r"voiceTriggers:\s*\[\s*\"", src):
                out += _Parser(src, src.index("[", m.start())).value()
    return out


def commands_v2_match(text):
    """CommandV2Engine.matchTrigger, CommandV2Engine.swift:569-649, on a fresh install: a `{project}` slot names no
    known project (ProjectsResolver.namesAKnownProject, IOSDispatcherV2.swift:1678-1700: no alias, no registry, no
    ~/Projects), any other slot needs a command-shaped literal prefix (two words or a trailing ':' or '-')."""
    q = text.lower().strip()
    if not q:
        return None
    for trig in commands_v2_triggers(src_v331()):
        t = trig.lower()
        if "{" not in t:
            if q == t:
                return trig
            continue
        rx, slots, i = "^", [], 0
        while i < len(t):
            if t[i] == "{" and "}" in t[i:]:
                end = t.index("}", i)
                slots.append(t[i + 1:end])
                rx += "(.+)"
                i = end + 1
                continue
            rx += "\\" + t[i] if t[i] in ".\\+*?()[]^$|" else t[i]
            i += 1
        if not re.match(rx + "$", q):
            continue
        prefix = t[:t.index("{")].strip()
        shaped = prefix.endswith(":") or prefix.endswith("-") or len(prefix.split(" ")) >= 2
        if "project" not in slots and shaped:
            return trig
    return None


def not_sent_reason(asks):
    """The fast path that answers this message with no model, in send()'s own order, or None."""
    return next((k for k in FAST_PATHS if asks["fast"] == k or (k == "pim" and asks["pim"])
                 or (k == "greeting" and asks["greeting"]) or (k == "clarify" and asks["confused"])), None)


# --------------------------------------------------------------------------- the 3.3.1 turn

def assemble_v331(text, model_id, memory_gb=16, asks=None):
    """ChatService.assemblePendingContext's lean branch, ChatService.swift:1092-1208, for a one-message chat with no
    preset, no image and no tool run in the last fifteen minutes. Returns the system blocks, the messages, the
    offered tools (wire order), which list the turn took and why, and what was cut."""
    asks = asks or asks_v331(text)
    window = context_tokens(memory_gb)                       # LocalTurn.contextTokens, LocalTurn.swift:23-29
    blocks = system_blocks_v331(model_id, asks)
    every = all_tools_v331()
    tools = relevance_filter_v331(every, text)               # ChatService.swift:1108-1112
    messages = [{"role": "user", "content": text}]           # ChatService.modelFacingMessages, ChatService.swift:1706-1728
    no_tools = asks["no_tools"]                              # ChatService.swift:1126-1127
    if no_tools:
        tools = []
    filtered = tools
    tools = rank_v331(tools, text)                           # ChatService.swift:1130
    chips = [] if no_tools else chip_tools(text)             # ChatService.swift:1137-1141
    if chips:
        lead = [next(x for x in tools if x["name"] == n) for n in chips if any(x["name"] == n for x in tools)]
        tools = lead + [t for t in tools if t["name"] not in chips]
    worst = cut_tools_line([t["name"] for t in tools])       # ChatService.swift:1143
    cached = [b for b in blocks if "cache_control" in b]
    msg_chars = [message_chars(m) for m in messages]
    msg_chars[-1] += _block_chars([b for b in blocks if "cache_control" not in b]) + (_count(worst) if worst else 0)
    n_msgs, n_tools = fit_v331(_block_chars(cached), msg_chars, [_tool_chars(t) for t in tools],
                               min(window, PROMPT_CEILING))  # ChatService.swift:1154-1156
    messages = messages[history_start(1, n_msgs):]           # ChatService.swift:1157-1158
    order = canonical_order(every)                           # ChatService.swift:1161
    why = []
    stable = None
    if no_tools:
        why.append("no_tools")
    elif chips:
        why.append("chip")
    else:
        stable = stable_local_tools(blocks, messages, tools, text, min(window, PROMPT_CEILING), window, why)
    if stable is not None:                                   # ChatService.swift:1166-1178
        names = {t["name"] for t in stable}
        cut = cut_tools_line([t["name"] for t in in_canonical_order([t for t in filtered if t["name"] not in names],
                                                                     order)])
        offered = stable
    else:                                                    # ChatService.swift:1179-1185
        cut = cut_tools_line([t["name"] for t in in_canonical_order(tools[n_tools:], order)])
        offered = in_canonical_order(tools[:n_tools], order)
    if cut:
        blocks = blocks + [{"type": "text", "text": cut}]
    prefill = asks["prefill"]                                # ChatService.swift:1188-1190
    if prefill is not None:
        messages = messages + [{"role": "assistant", "content": prefill}]
    return {"system_blocks": blocks, "messages": messages, "tools": for_model(model_id, offered),
            "offered": [t["name"] for t in offered], "filtered": [t["name"] for t in filtered],
            "ranked": [t["name"] for t in tools], "fit": [n_msgs, n_tools], "stable": stable is not None,
            "why_ranked": why[0] if why else None, "cut": cut, "prefill": prefill, "window": window}


def _sorted(obj):
    """Every dictionary with its keys sorted, as OpenAICompatBackend.wireData sends them (OpenAICompatBackend.swift:
    212-214), so the request's own bytes go out in the app's order."""
    if isinstance(obj, dict):
        return {k: _sorted(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sorted(x) for x in obj]
    return obj


def wire_messages_v331(blocks, messages):
    """OpenAICompatBackend.withTurnContextAfterTools, OpenAICompatBackend.swift:252-272: the cached blocks are the
    system message and the rest open the newest user message under the turn label."""
    def joined(bs):
        return "\n\n".join(b["text"] for b in bs if b["text"])
    stable = joined([b for b in blocks if "cache_control" in b])
    per_turn = joined([b for b in blocks if "cache_control" not in b])
    last = max(i for i, m in enumerate(messages) if m["role"] == "user")
    out = [dict(m) for m in messages]
    out[last]["content"] = f"{suite.TURN_LABEL}\n{per_turn}\n\nTHE USER'S MESSAGE:\n{out[last]['content']}"
    return [{"role": "system", "content": stable}] + out


def chat_request_v331(text, model, memory_gb=16):
    """The request Grux 3.3.1 sends Ollama for a one-message local chat turn, or, when the app answers the message
    itself, no request: {"not_sent": <fast path>}. num_ctx is this Mac's window (LocalTurn.contextTokens), seed 7 is
    the eval's own (suite.SEED), and every other field is the app's (OpenAICompatBackend.ollamaChatBody,
    OpenAICompatBackend.swift:290-309; streamedChatBody, OpenAICompatBackend.swift:552-563)."""
    # A waiting workflow's gate (ChatService.swift:215) needs a run in flight: a fresh install has none.
    if commands_v2_match(text):                              # ChatService.swift:221-270
        return {"not_sent": "commands_v2"}
    asks = asks_v331(text)
    reason = not_sent_reason(asks)
    if reason:
        return {"not_sent": reason}
    turn = assemble_v331(text, model, memory_gb, asks)
    body = {"model": model, "messages": wire_messages_v331(turn["system_blocks"], turn["messages"]), "stream": False,
            "think": False, "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": turn["window"], "num_predict": CHAT_MAX_TOKENS, "temperature": 0.5,
                        "seed": suite.SEED}}
    if turn["tools"]:
        body["tools"] = [wire(t) for t in turn["tools"]]
    return {"path": "/api/chat", "body": _sorted(body), "offered": turn["offered"], "prefill": turn["prefill"],
            "stable": turn["stable"], "why_ranked": turn["why_ranked"], "cut": turn["cut"]}


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
