#!/bin/zsh
export PATH=/opt/homebrew/bin:$PATH
L=~/grux-eval/run-nb256.log; echo "$(date) start" >> $L
cd ~/grux-eval/nb128 && printf "FROM grux-local-2b-r5\nPARAMETER num_batch 256\n" > Modelfile256 && ollama create grux-local-2b-r5-nb256 -f Modelfile256 > /dev/null 2>&1; echo "$(date) created rc=$?" >> $L
M=grux-local-2b-r5-nb256
(cd ~/grux-eval/src-main/recipe && GRUX_SRC=$HOME/grux-eval/src/Grux GRUX_SRC_V3=$HOME/grux-eval/src-local/Grux331 python3 eval_local.py --model $M --run ~/grux-eval/runs/nb256-8k --set local --num-ctx 8192 > /dev/null 2>&1); echo "$(date) local8k rc=$?" >> $L
(cd ~/grux-eval/fname-ab && NUM_CTX=8192 python3 ab.py $M > fn8k-$M.txt 2>&1); echo "$(date) fn8k rc=$?" >> $L
echo NB256-ALLDONE >> $L
