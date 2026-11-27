#!/bin/zsh
export PATH=/opt/homebrew/bin:$PATH
M=grux-local-2b-r5-nb256; L=~/grux-eval/run-nb256.log
E() { (cd ~/grux-eval/src-main/recipe && GRUX_SRC=$HOME/grux-eval/src/Grux GRUX_SRC_V3=$HOME/grux-eval/src-local/Grux331 python3 eval_local.py --model $M "$@"); }
for s in holdout holdout2; do E --run ~/grux-eval/runs/nb256 --set $s > /dev/null 2>&1; echo "$(date) $s rc=$?" >> $L; done
E --run ~/grux-eval/runs/nb256 > /dev/null 2>&1; echo "$(date) s1 rc=$?" >> $L
echo NB256B-ALLDONE >> $L
