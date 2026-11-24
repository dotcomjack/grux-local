#!/bin/sh
# Compile the oracle: Grux v3.3.1's own Swift for the local turn and the asks it answers itself, as is (prep.py cuts
# only the functions that read live app state), plus stubs.swift and main.swift.
#   sh recipe/grux_oracle/build.sh <out dir>        (GRUX_SRC_V3 points at Grux-Mac/Sources/Grux at v3.3.1)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$1"
S="${GRUX_SRC_V3:?export GRUX_SRC_V3 (GRUX_TAG=v3.3.1 sh recipe/get_grux_source.sh)}"
rm -rf "$OUT/src"
mkdir -p "$OUT/src"
python3 "$HERE/prep.py" "$S" "$OUT/src" Chat/LocalTurn.swift Chat/ToolRelevance.swift Chat/IntentClassifier.swift \
  Chat/DraftAsk.swift Chat/ListAsk.swift Chat/TrainedToolsV4.swift Chat/CapabilityAsk.swift Chat/PrivacyAsk.swift \
  "Chat/PrivacyAsk+Kinds.swift" Chat/VoiceAsk.swift Chat/SelfAsk.swift Chat/ToDoListAsk.swift Chat/MathAsk.swift \
  Chat/DayCountAsk.swift Chat/DateAsk.swift Chat/ForeignNoteAsk.swift Chat/TimeAsk.swift Chat/CalendarAsk.swift \
  Chat/CalendarSearchAsk.swift Chat/WeatherAsk.swift Chat/NeverMindAsk.swift Chat/GreetingAsk.swift \
  Decisions/DecisionModel.swift Voice/PIMIntents.swift Jax/Cognition/ConfidenceGate.swift \
  Jax/Cognition/ConfidenceGateGrounding.swift Jax/Grounding/ProductCatalogSeed.swift
# TrainedTools.swift up to forModel (its contract and samples read ChatService), and GruxReminder alone.
sed -n 1,124p "$S/Chat/TrainedTools.swift" > "$OUT/src/TrainedTools.swift"
echo "}" >> "$OUT/src/TrainedTools.swift"
sed -n 1,109p "$S/Reminders/ReminderState.swift" > "$OUT/src/ReminderModel.swift"
swiftc -O -parse-as-library -o "$OUT/oracle" "$HERE/stubs.swift" "$HERE/main.swift" "$OUT"/src/*.swift 2> "$OUT/build.log" \
  || { grep "error:" "$OUT/build.log" | head -20; exit 1; }
