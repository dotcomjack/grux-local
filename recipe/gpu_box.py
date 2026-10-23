"""One ephemeral Daytona GPU box serving Ollama, reached through Daytona's preview proxy.

Nothing of ours goes into the box: it only pulls public Ollama tags and answers HTTP. The
harness that sends the (synthetic) prompts runs outside it, so every result is already out of
the box the moment it is scored; a spot eviction loses at most the request in flight.
The Daytona key stays in the environment (DAYTONA_API_KEY=<your key> ...)."""
import json
import time
import urllib.error
import urllib.request

from daytona import (CreateSandboxFromImageParams, Daytona, GpuType, Image, ListSandboxesQuery,
                     Resources, SessionExecuteRequest)

from lease import LEASE_MINUTES, Lease

PORT = 11434


class OllamaBox:
    def __init__(self, gpu_types, label, ttl_minutes, spot=True, log=print, num_parallel=4, allow_paid=False):
        self.gpu_types = [GpuType(t) if isinstance(t, str) else t for t in gpu_types]
        self.label = {"grux.factory": label}
        self.ttl = ttl_minutes
        self.spot = spot
        # The free credit applies to "GPU Spot Sandbox" only (wallet breakdown, 2026-10-22); an
        # on-demand GPU box bills the PAID balance. So on-demand is never a silent fallback.
        self.allow_paid = allow_paid
        self.log = log
        self.d = Daytona()
        self.sb = None
        self.url = self.token = None
        self.num_parallel = num_parallel
        self.info = {}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        t0 = time.time()
        attempts = [True] + ([False] if self.allow_paid else [])
        if not self.spot and not self.allow_paid:
            raise RuntimeError("an on-demand GPU box bills real money; pass allow_paid=True with the owner's yes")
        errors = []
        for spot in attempts:
            try:
                self.sb = self.d.create(CreateSandboxFromImageParams(
                    image=Image.base("ollama/ollama:latest"),
                    resources=Resources(gpu=1, gpu_type=self.gpu_types),
                    env_vars={"OLLAMA_HOST": f"0.0.0.0:{PORT}", "OLLAMA_KEEP_ALIVE": "30m",
                              "OLLAMA_NUM_PARALLEL": str(self.num_parallel), "OLLAMA_MAX_LOADED_MODELS": "1"},
                    ephemeral=True, spot=spot, ttl_minutes=LEASE_MINUTES, auto_stop_interval=30,
                    labels={**self.label, "dcj.owner": "grux", "dcj.max-minutes": str(self.ttl)}),
                    timeout=900)
                break
            except Exception as e:  # spot fails at once when there is no spare capacity
                errors.append(f"spot={spot}: {str(e)[:200]}")
        if self.sb is None:
            raise RuntimeError("no GPU box: " + " | ".join(errors))
        self.lease = Lease(self.d, self.sb)
        m = self.sb.model_dump()
        self.info = {"id": m.get("id"), "gpu_type": m.get("gpu_type"), "gpu": m.get("gpu"),
                     "target": m.get("target"), "spot": m.get("spot"), "cpu": m.get("cpu"),
                     "memory": m.get("memory"), "disk": m.get("disk"), "create_errors": errors,
                     "created_s": round(time.time() - t0, 1), "started_at": time.time()}
        self.sb.process.create_session("ollama")
        self.sb.process.execute_session_command(
            "ollama", SessionExecuteRequest(command="ollama serve", run_async=True))
        link = self.sb.get_preview_link(PORT)
        self.url, self.token = link.url.rstrip("/"), link.token
        for _ in range(60):
            try:
                self.get("/api/version")
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("ollama never answered through the preview proxy")
        smi = self.sb.process.exec("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        self.info["nvidia_smi"] = (smi.result or "").strip()
        self.log(f"box up: {self.info}")

    def gpu_used_gb(self):
        """What the GPU itself reports in use, the ground truth when Ollama's /api/ps undercounts
        (2026-10-22: /api/ps read gemma4:26b, an 18 GB download, at 1.36 GB)."""
        r = self.sb.process.exec("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
        try:
            return round(float((r.result or "0").strip().splitlines()[0]) / 1024, 2)
        except (ValueError, IndexError):
            return None

    def evicted(self):
        try:
            self.sb.refresh_data()
            return bool(self.sb.model_dump().get("spot_evicted_at"))
        except Exception:
            return True

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

    def pull(self, model, timeout=3600):
        """A failed pull is a result, never a crash: (status, seconds), status carrying the
        server's own words. Measured 2026-10-22: one pull answered HTTP 500 and the
        uncaught error ended a whole pass."""
        t0 = time.time()
        for attempt in (1, 2):
            try:
                out = self.post("/api/pull", {"model": model, "stream": False}, timeout=timeout)
                return out.get("status"), round(time.time() - t0, 1)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                status = f"error HTTP {e.code}: {body}"
            except Exception as e:
                status = f"error {type(e).__name__}: {str(e)[:200]}"
            self.log(f"{model}: pull attempt {attempt} failed: {status}")
            time.sleep(10)
        return status, round(time.time() - t0, 1)

    def drop(self, model):
        """Free disk after a model is scored (512 GB per GPU, 241 GB for the whole catalog)."""
        try:
            self._req("DELETE", "/api/delete", {"model": model}, timeout=120)
        except Exception:
            pass

    def stop(self):
        if self.sb is not None:
            self.info["box_seconds"] = round(time.time() - self.info.get("started_at", time.time()), 1)
            try:
                self.lease.release()
            except Exception as e:
                self.log(f"delete error: {e} (the {LEASE_MINUTES} minute TTL still ends it)")
            self.sb = None
        time.sleep(5)
        self.info["boxes_left"] = [s.id for s in self.d.list(ListSandboxesQuery(labels=self.label))]
        self.log(f"box down: seconds {self.info.get('box_seconds')}, left {self.info['boxes_left']}")
