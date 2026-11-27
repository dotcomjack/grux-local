#!/bin/zsh
# Published nb256 (manifest 86255f20): fresh and after-use size at num_ctx 16384 around held-out v1, the local set at 16384,
# then tools/measure-speed.py for all sizes with gruxai.com/grux/grux-local:2b = 86255f20.
export PATH=/opt/homebrew/bin:$PATH
M=grux-local-2b-r5-nb256; O=~/grux-eval/runs/nb256-published; L=~/grux-eval/measure-nb256.log; mkdir -p $O
E() { (cd ~/grux-eval/src-main/recipe && GRUX_SRC=$HOME/grux-eval/src/Grux GRUX_SRC_V3=$HOME/grux-eval/src-local/Grux331 python3 eval_local.py --model $M "$@"); }
unload() { for m in $(curl -s localhost:11434/api/ps | python3 -c "import json,sys;[print(x['name']) for x in json.load(sys.stdin)['models']]"); do curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null; done; sleep 5; }
unload; curl -s localhost:11434/api/ps > $O/ps-empty.json
curl -s localhost:11434/api/generate -d "{\"model\":\"$M\",\"prompt\":\"hi\",\"stream\":false,\"think\":false,\"options\":{\"num_ctx\":16384,\"num_predict\":1}}" >/dev/null
curl -s localhost:11434/api/ps > $O/ps-fresh.json
E --run $O --set holdout > $O/eval-holdout.log 2>&1; echo "$(date) holdout rc=$?" >> $L
curl -s localhost:11434/api/ps > $O/ps-after.json
P=$(ps -axo pid=,args= | awk '/llama-server/ && /58b9d3c2/ && !/awk/{print $1}' | head -1); footprint -p $P > $O/footprint-after.txt 2>&1
E --run $O --set local > /dev/null 2>&1; echo "$(date) local16k rc=$?" >> $L
unload; ollama cp $M gruxai.com/grux/grux-local:2b >> $L 2>&1
cd ~/grux-eval/speed-run && python3 tools/measure-speed.py > run-nb256.log 2>&1; echo "$(date) speed rc=$?" >> $L
cp data/speed.json $O/speed.json
echo MEASURE-NB256-DONE >> $L
