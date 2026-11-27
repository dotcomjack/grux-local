#!/bin/zsh
export PATH=/opt/homebrew/bin:$PATH
OLLAMA_HOST=127.0.0.1:11501 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve > /tmp/batch.log 2>&1 &
SP=$!; sleep 4
python3 -I ~/grux-eval/batch_probe.py grux-local-2b-r5 0 512 256 128 64
kill $SP; sleep 2; echo "stopped $SP"
