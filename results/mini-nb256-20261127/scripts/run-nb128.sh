#!/bin/zsh
# num_batch 128 candidate (manifest f8542525) on every set the pilot ran, one at a time, recipe main 419a6bd.
export PATH=/opt/homebrew/bin:$PATH
M=grux-local-2b-r5-nb128; L=~/grux-eval/run-nb128.log
E() { (cd ~/grux-eval/src-main/recipe && GRUX_SRC=$HOME/grux-eval/src/Grux GRUX_SRC_V3=$HOME/grux-eval/src-local/Grux331 python3 eval_local.py --model $M "$@"); }
echo "$(date) start" >> $L
for s in holdout holdout2 local; do E --run ~/grux-eval/runs/nb128 --set $s > /dev/null 2>&1; echo "$(date) $s rc=$?" >> $L; done
E --run ~/grux-eval/runs/nb128 > /dev/null 2>&1; echo "$(date) s1 rc=$?" >> $L
E --run ~/grux-eval/runs/nb128-8k --set local --num-ctx 8192 > /dev/null 2>&1; echo "$(date) local8k rc=$?" >> $L
(cd ~/grux-eval/fname-ab && NUM_CTX=8192 python3 ab.py $M > fn8k-$M.txt 2>&1); echo "$(date) fn8k rc=$?" >> $L
echo NB128-ALLDONE >> $L
