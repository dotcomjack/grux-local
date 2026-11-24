import Foundation

struct ClaudeTool {
    let name: String
    let description: String
    let inputSchema: [String: Any]
}

enum ModelRates { static func host(fromBaseURL s: String) -> String { s } }

struct HardwareProfile {
    var physicalMemoryGB: Double
    static func detect() -> HardwareProfile { HardwareProfile(physicalMemoryGB: 16) }
}

// ShellRecoveryTools.swift:18-21 at v3.3.1, the constants only
enum ShellRecoveryTools {
    static let auditTool = "shell_audit_log"
    static let listTool = "shell_snapshots"
    static let restoreTool = "shell_restore_snapshot"
    static let names: Set<String> = [auditTool, listTool, restoreTool]
}

// Reply-side helpers only (DraftAsk.checkingDays, unchangedTransform, NeverMindAsk.reply); the oracle never calls them.
enum ActionClaimGuard {
    static let approvalsPrefix = ""
    static let notDoneYet = ""
    static func appending(_ l: String, to r: String) -> String { r }
    static func approvalsLine(_ w: [String]) -> String { "" }
}

struct ChatMessage {
    enum Role { case user, assistant }
    let role: Role
    let isNotice: Bool
    let imageData: Data?
}

// Reply-side app state the matchers' reply functions read; the oracle calls only the matchers.
enum ChatReadiness {
    case ready, needsModel, localModelFoundButNotRouted, offlinePinnedButNoLocalModel
    var canSend: Bool { true }
    static func current() -> ChatReadiness { .ready }
}
enum FeatureState { case ready, degraded, needsSetup, notChosen, unavailable }
enum FeatureRegistry {
    struct Row { let id: String; let label: String }
    static let rows: [Row] = []
    static func state(of r: Row) -> FeatureState { .ready }
}
final class CalendarService {
    static let shared = CalendarService()
    var hasAccess: Bool { false }
    func events(from: Date, to: Date) -> [EventSummary] { [] }
    struct EventSummary: Codable, Equatable, Hashable, Identifiable, Sendable {
        let id: String
        let title: String
        let start: Date
        let end: Date
        let isAllDay: Bool
        let location: String?
        let notes: String?
        let calendarName: String
        let calendarId: String
    }
}

enum ActiveProvider { case anthropic, local, custom }
enum CorpusPermission { case ready, needsFullDiskAccess, needsAutomation, needsAccount, needsPassword, needsKeychain, noData, unknown }
enum RAGClient { static let baseURLDefaultsKey = "rag" }

// DecisionEngine.swift stand-in: the oracle never asks the engine (a keyless install answers on device, confirmed).
final class DecisionEngine {
    static let shared = DecisionEngine()
    func decide(surface: String, state: String, questions: [String: DecisionQuestion]) async -> DecisionResult { fatalError() }
}
enum TaskPriority: String, Codable, CaseIterable, Identifiable {
    case now, next, later
    var id: String { rawValue }
    var label: String {
        switch self {
        case .now: return "NOW"
        case .next: return "NEXT"
        case .later: return "LATER"
        }
    }
}

enum GruxGlassTone { case urgent, normal }

enum UpdateCheck { static func sessionConfiguration() -> URLSessionConfiguration { .ephemeral } }

// The model path (ConfidenceGate.modelPathEnabled is false in the app); never reached here.
extension ConfidenceGate {
    static func assessWithModel(prompt: String, context: ConfidenceContext, heuristicScore: Double, apiKey: String, model: String) async -> ConfidenceVerdict { fatalError() }
}

// A fresh install: no ~/.grux/product-catalog.json, so the catalog is empty.
enum Persistence { static let gruxDir = URL(fileURLWithPath: "/nonexistent-grux-dir") }
