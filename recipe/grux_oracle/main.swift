import Foundation

// Oracle: Grux v3.3.1's own Swift (LocalTurn, ToolRelevance, DraftAsk, ListAsk, TrainedTools, the fast-path
// matchers) compiled as is, plus a verbatim copy of the lean branch of ChatService.assemblePendingContext
// (ChatService.swift:1119-1192) and ChatService.stableLocalTools (ChatService.swift:1226-1253) for a one-message
// turn with no preset, no recent tools and no image. Reads a JSON job, writes a JSON answer.

func opt(_ s: String?) -> Any { s.map { $0 as Any } ?? NSNull() }

func chars(_ blocks: [[String: Any]]) -> Int {
    blocks.compactMap { $0["text"] as? String }.reduce(0) { $0 + $1.count }
}

func allTools(_ raw: [[String: Any]]) -> [ClaudeTool] {
    raw.map { ClaudeTool(name: $0["name"] as! String, description: $0["description"] as! String,
                         inputSchema: $0["schema"] as! [String: Any]) }
}

var ALL: [ClaudeTool] = []

func stableLocalTools(systemBlocks: [[String: Any]], messages: [[String: Any]], ranked: [ClaudeTool],
                      utterance: String, sizingWindow: Int, contextWindow: Int) -> [ClaudeTool]? {
    guard !ranked.contains(where: { ToolRelevance.group(for: $0.name) == .otherApps }) else { return nil }
    let every = ALL
    let stable = LocalTurn.stableTools(core: every.filter { ToolRelevance.group(for: $0.name) == .core },
                                       cachedSystemChars: chars(Array(systemBlocks.prefix(1))),
                                       longestCutLineChars: LocalTurn.cutToolsLine(every.map(\.name))?.count ?? 0,
                                       window: sizingWindow)
    let names = Set(stable.map(\.name))
    guard !stable.isEmpty,
          !LocalTurn.pointsPast(stable, candidates: ranked, utterance: utterance) else { return nil }
    let cutLine = LocalTurn.cutToolsLine(ranked.map(\.name).filter { !names.contains($0) })?.count ?? 0
    let prompt = LocalTurn.tokens(chars: chars(systemBlocks) + cutLine)
        + messages.map(LocalTurn.chars).map(LocalTurn.tokens).reduce(0, +)
        + stable.map(LocalTurn.chars).map(LocalTurn.tokens).reduce(0, +)
    guard prompt + LocalTurn.replyTokens + 256 <= contextWindow else { return nil }
    return stable
}

func assemble(_ text: String, systemBlocks inBlocks: [[String: Any]], window: Int) -> [String: Any] {
    var systemBlocks = inBlocks
    let ceiling = LocalTurn.promptCeiling
    var tools = ToolRelevance.filter(ALL, signals: .init(utterance: text))
    var messages: [[String: Any]] = [["role": "user", "content": text]]
    let visibleCount = 1
    var stablePath = false
    let noTools = DraftAsk.matches(text) || ListAsk.matches(text) || DraftAsk.transformsPastedText(text)
    if noTools { tools = [] }
    let filtered = tools
    tools = LocalTurn.rank(tools, utterance: text)
    let chipTools = noTools ? [] : ToolRelevance.chipTools(for: text)
    if !chipTools.isEmpty {
        let lead = chipTools.compactMap { name in tools.first { $0.name == name } }
        tools = lead + tools.filter { !chipTools.contains($0.name) }
    }
    let rankedNames = tools.map(\.name)   // the list the fit reads, chip tools first
    let worstLine = LocalTurn.cutToolsLine(tools.map(\.name))?.count ?? 0
    let stableChars = chars(systemBlocks.filter { $0["cache_control"] != nil })
    var messageChars = messages.map(LocalTurn.chars)
    if !messageChars.isEmpty {
        messageChars[messageChars.count - 1] += chars(systemBlocks.filter { $0["cache_control"] == nil }) + worstLine
    }
    let fit = LocalTurn.fit(systemChars: stableChars, messageChars: messageChars,
                            toolChars: tools.map(LocalTurn.chars), window: min(window, ceiling))
    messages = Array(messages.suffix(visibleCount - LocalTurn.historyStart(count: visibleCount, fitting: fit.messages)))
    let order = LocalTurn.canonicalOrder(ALL)
    var cutText: String? = nil
    if !noTools, chipTools.isEmpty,
       let stable = stableLocalTools(systemBlocks: systemBlocks, messages: messages, ranked: tools, utterance: text,
                                     sizingWindow: min(window, ceiling), contextWindow: window) {
        let names = Set(stable.map(\.name))
        let cut = LocalTurn.inCanonicalOrder(filtered.filter { !names.contains($0.name) }, order: order)
        if let line = LocalTurn.cutToolsLine(cut.map(\.name)) {
            systemBlocks.append(["type": "text", "text": line])
            cutText = line
        }
        tools = stable
        stablePath = true
    } else {
        let cut = LocalTurn.inCanonicalOrder(Array(tools.dropFirst(fit.tools)), order: order)
        if let line = LocalTurn.cutToolsLine(cut.map(\.name)) {
            systemBlocks.append(["type": "text", "text": line])
            cutText = line
        }
        tools = LocalTurn.inCanonicalOrder(Array(tools.prefix(fit.tools)), order: order)
    }
    var prefill: Any = NSNull()
    if let start = DraftAsk.prefillMessage(ask: text) ?? ListAsk.prefillMessage(ask: text) {
        messages.append(start)
        prefill = start["content"] as? String ?? NSNull()
    }
    return ["filtered": filtered.map(\.name), "ranked": rankedNames, "chip": chipTools, "no_tools": noTools,
            "fit": [fit.messages, fit.tools], "stable": stablePath, "tools": tools.map(\.name),
            "cut": opt(cutText), "prefill": prefill, "messages": messages.count]
}

@MainActor
func fastPath(_ trimmed: String, now: Date) -> String? {
    if ChatIntentClassifier.isUpgradeYourselfAsk(trimmed) { return "upgrade_yourself" }
    if CapabilityAsk.matches(trimmed, earlier: [], name: "Jax") { return "capability" }
    if PrivacyAsk.matches(trimmed, earlier: []) { return "privacy" }
    if PrivacyAsk.kind(of: trimmed) != nil { return "privacy_kind" }
    if VoiceAsk.matches(trimmed) { return "voice" }
    if SelfAsk.kind(of: trimmed, earlier: [], name: "Jax") != nil { return "self" }
    if ToDoListAsk.matches(trimmed) { return "todo_list" }
    if MathAsk.reply(for: trimmed) != nil { return "math" }
    if DayCountAsk.reply(for: trimmed, now: now) != nil { return "day_count" }
    if DateAsk.reply(for: trimmed, afterImage: false, now: now) != nil { return "date" }
    if ForeignNoteAsk.reply(for: trimmed) != nil { return "foreign_note" }
    if TimeAsk.match(trimmed) != nil { return "time" }
    if CalendarAsk.window(for: trimmed, now: now) != nil { return "calendar" }
    if CalendarSearchAsk.match(trimmed) != nil { return "calendar_search" }
    if WeatherAsk.match(trimmed) ?? WeatherAsk.followUp(trimmed, previousReply: nil, previousAsk: nil)
        ?? WeatherAsk.dayFollowUp(trimmed, previousReply: nil, remembered: nil) != nil { return "weather" }
    if NeverMindAsk.matches(trimmed) { return "never_mind" }
    return nil
}

@main
struct Oracle {
    @MainActor static func main() throws {
        let job = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
            as! [String: Any]
        ALL = allTools(job["tools"] as! [[String: Any]])
        let now = Date(timeIntervalSince1970: job["now"] as! Double)
        var turns: [[String: Any]] = []
        for t in job["turns"] as! [[String: Any]] {
            let text = t["utterance"] as! String
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let blocks = (t["system_blocks"] as! [[String: Any]]).map { b -> [String: Any] in
                var o: [String: Any] = ["type": "text", "text": b["text"] as! String]
                if b["cached"] as? Bool == true { o["cache_control"] = ["type": "ephemeral"] }
                return o
            }
            var out = assemble(text, systemBlocks: blocks, window: t["window"] as! Int)
            out["fast"] = opt(fastPath(trimmed, now: now))
            out["greeting"] = GreetingAsk.matches(trimmed, name: "Jax")
            out["pim"] = opt(ChatIntentClassifier.pimRoute(utterance: trimmed, now: now).map { "\($0.kind.rawValue)" })
            let verdict = ConfidenceGate.assessGroundedHeuristic(prompt: trimmed, context: ConfidenceContext(
                hasPriorTurns: false, lastTurnText: "", hasOnScreenSubject: false, brand: nil))
            out["confused"] = verdict.isTrulyConfused && verdict.clarifyingQuestion != nil
            let v = ChatIntentClassifier.classify(utterance: text, hasImage: false)
            out["sections"] = [v.taskStack, v.proposedActions, v.recentFocus, v.availableMacros]
            out["notes"] = ["weather": opt(WeatherAsk.localNote(ask: text, remembered: nil)),
                            "draft": opt(DraftAsk.localNote(ask: text, now: now)),
                            "list": opt(ListAsk.localNote(ask: text)),
                            "pasted": opt(DraftAsk.pastedTextNote(ask: text))]
            turns.append(out)
        }
        let tools = ALL
        let global: [String: Any] = [
            "stable_head": LocalTurn.stableHead,
            "signatures": Dictionary(uniqueKeysWithValues: tools.map { ($0.name, TrainedTools.signature($0)) }),
            "hashes": Dictionary(uniqueKeysWithValues: tools.map { ($0.name, TrainedTools.schemaHash($0)) }),
            "trained": tools.filter(TrainedTools.isTrained).map(\.name),
            "canonical": LocalTurn.stableOrder(tools).map(\.name),
            "chars": tools.map(LocalTurn.chars),
        ]
        let data = try JSONSerialization.data(withJSONObject: ["turns": turns, "global": global], options: [.sortedKeys])
        FileHandle.standardOutput.write(data)
    }
}
