"""A spot GPU box with PyTorch for training, held by the same Lease as every other box.

Long jobs run as async session commands and are polled, so a dropped connection never kills a
training run, and every artifact leaves the box through the SDK's file download (no keys of ours
ever go into the box, so the box cannot push anywhere itself).
"""
import json
import os
import time
import urllib.request

from daytona import (CreateSandboxFromImageParams, Daytona, GpuType, Image, Resources,
                     SessionExecuteRequest)

from lease import LEASE_MINUTES, Lease

TORCH_IMAGE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel"
OLLAMA_VERSION = "0.40.8"
# Everything a training box needs, measured 2026-10-22: the image has no curl; pip needs
# --break-system-packages; Ollama on Linux imports Qwen 3.5 safetensors only through its MLX runtime,
# a separate download (without it: "MLX runtime is not available"), and only as int4, int8, nvfp4,
# mxfp4 or mxfp8 ("q4_K_M" is refused). nvfp4 is what a Mac serves, so the GPU score and the Mac
# check measure the same weights. Qwen 3.5's linear-attention layers fall back to slow reference code
# without flash-linear-attention, which in turn refuses Hopper with triton under 3.7.1 (wrong gradients):
# with both, 8-example batches ran 21 s a step against 36 s (2B, 32 examples a step, losses within 0.3%).
BOOT = ("apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq curl zstd ca-certificates >/dev/null 2>&1 && "
        "pip install -q -U --break-system-packages 'transformers>=4.57' peft accelerate safetensors sentencepiece "
        "huggingface_hub >/dev/null 2>&1 && "
        # Kernels AFTER, never with -U: one combined upgrade let pip replace the image's torch 2.11 (cu128)
        # with 2.14 (cu130) and transformers stopped importing (measured 2026-10-22, nine boxes).
        "pip install -q --break-system-packages flash-linear-attention >/dev/null 2>&1 && "
        "pip install -q --break-system-packages --no-deps 'triton>=3.7.1' >/dev/null 2>&1 && "
        "curl -fsSL https://ollama.com/install.sh | "
        f"OLLAMA_VERSION={OLLAMA_VERSION} sh >/tmp/ollama-install.log 2>&1; "
        f"curl -fsSL -o /tmp/mlx.tar.zst https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/"
        "ollama-linux-amd64-mlx.tar.zst && zstd -dc /tmp/mlx.tar.zst | tar -x -C /usr/local && rm -f /tmp/mlx.tar.zst; "
        "(OLLAMA_HOST=0.0.0.0:11434 OLLAMA_KEEP_ALIVE=30m nohup ollama serve >/tmp/ollama.log 2>&1 &); sleep 6; "
        "mkdir -p /workspace; ollama --version && python -c 'import torch, transformers, peft, fla, triton; "
        "from transformers import AutoModelForCausalLM; "
        "assert torch.__version__.startswith(\"2.11.\"), torch.__version__; "
        "print(torch.__version__, transformers.__version__, peft.__version__, triton.__version__, fla.__version__)'")


class TrainBox:
    def __init__(self, gpu_types, label, max_minutes, log=print, image=TORCH_IMAGE):
        self.gpu_types = [GpuType(t) if isinstance(t, str) else t for t in gpu_types]
        self.label = {"grux.factory": label, "dcj.owner": "grux", "dcj.max-minutes": str(max_minutes)}
        self.image, self.log = image, log
        self.d = Daytona()
        self.sb = self.lease = None
        self.info = {}

    def __enter__(self):
        t0 = time.time()
        # Spot only: the free credit covers GPU Spot sandboxes and nothing else on GPU.
        self.sb = self.d.create(CreateSandboxFromImageParams(
            image=Image.base(self.image), resources=Resources(gpu=1, gpu_type=self.gpu_types),
            ephemeral=True, spot=True, ttl_minutes=LEASE_MINUTES, auto_stop_interval=30,
            labels=self.label), timeout=1500)
        self.lease = Lease(self.d, self.sb)
        m = self.sb.model_dump()
        self.info = {"id": m.get("id"), "gpu_type": str(m.get("gpu_type")), "spot": m.get("spot"),
                     "created_s": round(time.time() - t0, 1), "started_at": time.time()}
        self.info["nvidia_smi"] = self.sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")[1]
        self.log(f"train box up: {self.info}")
        return self

    def __exit__(self, *exc):
        if self.sb is not None:
            self.info["box_seconds"] = round(time.time() - self.info["started_at"], 1)
            try:
                self.lease.release()
            except Exception as e:
                self.log(f"delete error: {e} (the {LEASE_MINUTES} minute TTL still ends it)")
            self.log(f"train box down after {self.info['box_seconds']} s")
            self.sb = None

    def sh(self, cmd, timeout=600):
        r = self.sb.process.exec(cmd, timeout=timeout)
        return r.exit_code, (r.result or "").strip()

    def run_long(self, name, cmd, max_s=4 * 3600, poll_s=30, done_marker=None):
        """Run `cmd` detached in its own session; poll until it exits. Returns (exit, log tail)."""
        self.sb.process.create_session(name)
        r = self.sb.process.execute_session_command(
            name, SessionExecuteRequest(command=f"cd /workspace && {cmd} > /workspace/{name}.log 2>&1; "
                                                f"echo EXIT=$? >> /workspace/{name}.log", run_async=True))
        t0 = time.time()
        while time.time() - t0 < max_s:
            time.sleep(poll_s)
            code, tail = self.sh(f"tail -c 2000 /workspace/{name}.log", timeout=60)
            last = tail.splitlines()[-1] if tail else ""
            if last.startswith("EXIT="):
                return int(last[5:] or 1), tail
            self.log(f"{name}: {last[-160:]}")
        return 124, self.sh(f"tail -c 2000 /workspace/{name}.log")[1]

    def put(self, local, remote):
        self.sb.fs.upload_file(local, remote)

    def get(self, remote, local, timeout=3600):
        """Stream a file out of the box to local disk (never through memory: checkpoints are GBs)."""
        from daytona import FileDownloadRequest
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        res = self.sb.fs.download_files([FileDownloadRequest(source=remote, destination=local)], timeout=timeout)
        err = getattr(res[0], "error", None) if res else "no response"
        if err:
            raise RuntimeError(f"download {remote}: {err}")
        return os.path.getsize(local)

    def ollama(self, port=11434):
        """A client for the box's own Ollama through the preview proxy (BOOT binds 0.0.0.0)."""
        link = self.sb.get_preview_link(port)
        return OllamaClient(link.url.rstrip("/"), link.token)

    def gone(self):
        """True once the box is evicted or deleted (spot boxes go without warning)."""
        try:
            self.sb.refresh_data()
            m = self.sb.model_dump()
            return bool(m.get("spot_evicted_at")) or str(m.get("state", "")).lower() in ("destroyed", "destroying", "error")
        except Exception:
            return True


class OllamaClient:
    def __init__(self, url, token):
        self.url, self.token = url, token

    def _req(self, method, path, payload=None, timeout=600):
        data = json.dumps(payload).encode() if payload is not None else None
        r = urllib.request.Request(self.url + path, data=data, method=method, headers={
            "Content-Type": "application/json", "x-daytona-preview-token": self.token,
            "X-Daytona-Skip-Preview-Warning": "true"})
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read().decode() or "{}")

    def get(self, path, timeout=30):
        return self._req("GET", path, timeout=timeout)

    def post(self, path, payload, timeout=600):
        return self._req("POST", path, payload, timeout=timeout)
