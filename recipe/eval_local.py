#!/usr/bin/env python3
"""S1 on a real Mac: the same suite, sent one request at a time to this Mac's own Ollama, as a
person's Grux would. Records pass, latency and the Ollama runner's memory footprint, so a pick is
proven on the hardware it is for, not only on a GPU.

  python3 eval_local.py --model qwen3.5:4b --run results/mac-<stamp> [--think false] [--per-family N]
                        [--set holdout|holdout2|holdout_v3|holdout_v4|local]

--set holdout scores S2's frozen held-out set (s2/holdout.py) instead of the S1 suite; --set holdout2 scores
held-out version 2 (s2/holdout_v2.py), whose chat-shaped cases are sent the way a Grux 3.1.0 local turn sends
them (local_turn.py) and scored against the tools that turn offered. --set holdout_v3 scores held-out version 3
(s2/holdout_v3.py): the full Grux turn with every tool at its pinned tag (GRUX_SRC_V3), each job sent step by
step with the scripted results of the steps before it; num_ctx defaults to that set's own window. --set holdout_v4
scores held-out version 4 (s2/holdout_v4.py): new jobs, sent and scored by held-out v3's own runner and scorer.
--set local (output suffix -local) scores held-out v2's 265 chat turns sent the way the current Grux, v3.3.1, sends a
local turn (s2/holdout_local.py, local_turn.chat_request_v331; GRUX_SRC_V3 at v3.3.1), with v2's scorer against the
tools that turn offered; --num-ctx is the Mac's own window (8192, 16384 or 32768, default 16384) and decides the tools
as the app does. A message 3.3.1 answers with no model is written as status "not_sent" with the reason, outside the
pass rate.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suite  # noqa: E402

BASE = "http://127.0.0.1:11434"

# The run stops itself before it can take the Mac down (test_eval_local_guard.py). Measured 2026-11-14 on the
# Mini, 16 GB with 4 GB of disk free: held-out v3 at 65,536 ran it out of swap, macOS killed system services, the
# login session reset and every LaunchAgent stopped until a password was typed.
MEM_FREE_PCT_MIN = 10      # memory_pressure's own free percentage
PRESSURE_CRITICAL = 4      # kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
DISK_FREE_GB_MIN = 8       # room for macOS to grow swap into; the Mini had 4.2
LARGE_WINDOW = 16384       # the disk floor binds above this: 16,384 runs held on the Mini with 0 B of swap
GUARD_EVERY_S = 5
UNLOAD_TIMEOUT_S = 5       # a dying Ollama may never answer; the runner is killed by pid right after
PEAK = {"gb": None}        # the runner's largest footprint, sampled by the guard
ABORTING = threading.Event()   # set the moment a breach is seen: no request may reload the model after it
BEAT = {"t": None}         # the guard's last completed tick; a guard silent past GUARD_STALE_S stops the run
GUARD_STALE_S = 120
SAMPLE_EVERY = 6           # footprint walks the runner's memory map; sample it every 6th tick, never on the breach path


def parse_swap_free_mb(text):
    m = re.search(r"free = (\d+(?:\.\d+)?)([MG])", text or "")
    if not m:
        return None
    return round(float(m.group(1)) * (1024 if m.group(2) == "G" else 1), 2)


def parse_free_pct(text):
    m = re.search(r"memory free percentage: (\d+)%", text or "")
    return int(m.group(1)) if m else None


def parse_footprint_gb(text, key="phys_footprint"):
    """macOS footprint's phys_footprint line in GB. Resident size (ps rss) misses the GPU memory an MLX or Metal
    runner holds: measured 2026-11-14 on the Mini, the 9B read 0.15 GB rss against 7.13 GB footprint."""
    m = re.search(key + r":\s*(\d+(?:\.\d+)?)\s*(KB|MB|GB|B)\b", text or "")
    if not m:
        return None
    return round(float(m.group(1)) * {"B": 1e-9, "KB": 1e-6, "MB": 1e-3, "GB": 1}[m.group(2)], 2)


def parse_runner_pids(ps_text, model):
    """Pids of the Ollama runner serving exactly `model`, from `ps -axo pid=,args=` lines."""
    pids = []
    for line in (ps_text or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        args = parts[1].split()
        if (args and os.path.basename(args[0]) == "ollama" and "runner" in args and "--model" in args
                and args.index("--model") + 1 < len(args) and args[args.index("--model") + 1] == model):
            pids.append(int(parts[0]))
    return pids


def mem_verdict(pressure_level, free_pct):
    """None when the Mac can take more, else why it cannot. An unreadable number never trips alone."""
    if pressure_level is not None and pressure_level >= PRESSURE_CRITICAL:
        return f"kernel memory pressure critical (level {pressure_level})"
    if free_pct is not None and free_pct < MEM_FREE_PCT_MIN:
        return f"memory free {free_pct}% under the {MEM_FREE_PCT_MIN}% floor"
    return None


def disk_verdict(free_gb, num_ctx=None):
    if num_ctx is not None and num_ctx <= LARGE_WINDOW:
        return None
    if free_gb is not None and free_gb < DISK_FREE_GB_MIN:
        return f"disk free {free_gb} GB under {DISK_FREE_GB_MIN} GB: no room to swap into"
    return None


def _sh(*cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout).stdout
    except Exception:   # a reader that throws would kill the guard thread; an empty reading never trips
        return ""


def read_mem():
    """(kernel pressure level, free percentage), each None where this OS has no reader."""
    level = _sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip()
    return (int(level) if level.isdigit() else None), parse_free_pct(_sh("memory_pressure"))


def guard_tick(read, unload, abort_path, exit=os._exit, kill=lambda: []):
    """One look. On a breach: stop every new request, kill the runner by pid (frees the memory at once, even when
    Ollama is too starved to answer), then ask Ollama to unload so its bookkeeping agrees, write why, exit 7."""
    level, free_pct = read()
    why = mem_verdict(level, free_pct)
    if why is None:
        return
    ABORTING.set()
    unloaded, killed = "not tried", []
    try:
        try:   # nothing slow runs before the kill: a thrashing Mac is the case this exists for
            killed = kill()
        except Exception as e:
            killed = ["kill failed: " + str(e)[:120]]
        try:
            unload()
            unloaded = "ok"
        except Exception as e:
            unloaded = "failed: " + str(e)[:120]
        try:   # a full disk or a closed stderr must not stop the exit below
            with open(abort_path, "w") as f:
                json.dump({"why": why, "pressure_level": level, "free_pct": free_pct, "at": time.time(),
                           "unload": unloaded, "killed": killed, "peak_runner_footprint_gb": PEAK["gb"],
                           "swap_free_mb": parse_swap_free_mb(_sh("sysctl", "vm.swapusage", timeout=5))}, f)
            print("ABORT " + why, file=sys.stderr, flush=True)
        except Exception:
            pass
    finally:
        exit(7)


def runner_pids(model):
    return parse_runner_pids(_sh("ps", "-axo", "pid=,args="), model)


def kill_runner(model):
    killed = []
    for pid in runner_pids(model):
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except OSError:
            pass
    return killed


def runner_footprint_gb(model, key="phys_footprint"):
    """The runner's memory as macOS counts it (GPU memory included). None where footprint does not exist (Linux):
    there, meta's ollama_size_gb is the number to read. key="phys_footprint_peak" reads the runner's own high-water
    mark, which catches a spike between two guard ticks."""
    vals = [parse_footprint_gb(_sh("footprint", "-p", str(pid), timeout=5), key) for pid in runner_pids(model)]
    vals = [v for v in vals if v is not None]
    return round(sum(vals), 2) if vals else None


def peak_sample(model):
    return runner_footprint_gb(model, "phys_footprint_peak") or runner_footprint_gb(model)


def guard_step(model, abort_path, read=None, tick=0):
    """One guard tick, a peak sample every SAMPLE_EVERY ticks, and a heartbeat. Nothing in here may raise: a dead
    guard thread fails open."""
    try:
        guard_tick(read or read_mem, lambda: _post("/api/generate", {"model": model, "keep_alive": 0},
                                                   timeout=UNLOAD_TIMEOUT_S),
                   abort_path, kill=lambda: kill_runner(model))
        if tick % SAMPLE_EVERY == 0:
            gb = peak_sample(model)
            if gb is not None and (PEAK["gb"] is None or gb > PEAK["gb"]):
                PEAK["gb"] = gb
    except Exception as e:
        try:
            print("guard tick skipped: " + str(e)[:120], file=sys.stderr, flush=True)
        except Exception:
            pass
    BEAT["t"] = time.time()


def start_guard(model, abort_path):
    def loop():
        tick = 0
        while True:
            guard_step(model, abort_path, tick=tick)
            tick += 1
            time.sleep(GUARD_EVERY_S)
    BEAT["t"] = time.time()
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def guard_alive():
    return BEAT["t"] is not None and time.time() - BEAT["t"] < GUARD_STALE_S


def hold_if_aborting(limit_s=60, exit=os._exit):
    """The guard is between its kill and its exit: write nothing (a 'skipped' row would never be retried). If the
    guard never gets to its exit (a hung write on a starved disk), stop the run here with the same code."""
    t0 = time.time()
    while ABORTING.is_set():
        if time.time() - t0 > limit_s:
            exit(7)
            return
        time.sleep(0.2)


def load_done(path):
    """Case ids already in a results file. A line cut off by an abort mid-write is skipped, and the file is made to
    end in a newline so the next row starts on its own line."""
    have = set()
    if not os.path.exists(path):
        return have
    text = open(path).read()
    for line in text.splitlines():
        try:
            have.add(json.loads(line)["case"])
        except (ValueError, KeyError, TypeError):
            pass
    if text and not text.endswith("\n"):
        with open(path, "a") as f:
            f.write("\n")
    return have


def _post(path, body, timeout=600):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def post(path, body, timeout=600):
    """Every request the run makes. Refused once the guard has seen a breach, so nothing reloads the model."""
    if ABORTING.is_set():
        raise RuntimeError("aborting: the memory guard tripped")
    return _post(path, body, timeout)


def main():
    """A breach can land during the model load, outside any per-case try: wait for the guard to finish its abort
    file and exit 7 rather than dying with exit 1 and no record."""
    try:
        _main()
    except BaseException:
        hold_if_aborting()
        raise


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--think", choices=["unset", "false"], default="false")
    ap.add_argument("--per-family", type=int, default=0, help="first N cases per job (0 = all)")
    ap.add_argument("--num-ctx", type=int, default=None, help="16384, or held-out v3's own window for holdout_v3")
    ap.add_argument("--set", choices=["s1", "holdout", "holdout2", "holdout_v3", "holdout_v4", "local"], default="s1")
    a = ap.parse_args()
    request, score, cases_fn, suffix = suite.request, suite.score, suite.cases, ""
    if a.set == "local":   # held-out v2's chat turns as Grux 3.3.1 sends them; num_ctx is this Mac's window
        sys.path.insert(0, os.path.join(HERE, "s2"))
        import holdout_local
        gb = holdout_local.memory_for(a.num_ctx or 16384)
        request = lambda c, model: holdout_local.request_local(c, model, gb)   # noqa: E731
        score, cases_fn, suffix = holdout_local.score_local, holdout_local.cases, "-local"
    if a.set == "holdout":
        sys.path.insert(0, os.path.join(HERE, "s2"))
        import holdout
        request, score, cases_fn, suffix = holdout.request_holdout, holdout.score_holdout, holdout.cases, "-holdout"
    if a.set == "holdout2":
        sys.path.insert(0, os.path.join(HERE, "s2"))
        import holdout_v2
        request, score, cases_fn, suffix = holdout_v2.request_v2, holdout_v2.score_v2, holdout_v2.cases, "-holdout2"
    v3 = None
    if a.set in ("holdout_v3", "holdout_v4"):   # v4's jobs go through v3's runner and scorer, unchanged
        sys.path.insert(0, os.path.join(HERE, "s2"))
        import holdout_v3 as v3
        cases_fn, suffix = v3.cases, "-holdout_v3"
        if a.set == "holdout_v4":
            import holdout_v4
            cases_fn, suffix = holdout_v4.cases, "-holdout_v4"
        if a.num_ctx is not None:
            v3.NUM_CTX = a.num_ctx
        a.num_ctx = v3.NUM_CTX
    if a.num_ctx is None:
        a.num_ctx = 16384
    if a.think == "false":
        suite.THINK = False
    suite.NUM_CTX = a.num_ctx
    os.makedirs(a.run, exist_ok=True)
    cases = cases_fn()
    if a.per_family:
        seen = {}
        cases = [c for c in cases if seen.setdefault(c["family"], []).append(1) or len(seen[c["family"]]) <= a.per_family]
    stem = a.model.replace(":", "_").replace("/", "_")   # a host-qualified name holds slashes
    path = os.path.join(a.run, stem + suffix + ".jsonl")
    have = load_done(path)
    free_gb = min(shutil.disk_usage("/").free, shutil.disk_usage(a.run).free) / 1e9   # swap grows on the boot volume
    refuse = disk_verdict(round(free_gb, 1), a.num_ctx) or mem_verdict(*read_mem())
    if refuse:
        sys.exit("REFUSED " + refuse)
    abort_path = os.path.join(a.run, stem + suffix + ".abort.json")
    if os.path.exists(abort_path):
        os.remove(abort_path)   # a reason from an earlier run must not sit beside this one's result
    start_guard(a.model, abort_path)
    t0 = time.time()
    warm = post("/api/chat", {"model": a.model, "stream": False, "think": False,
                              "messages": [{"role": "user", "content": "ok"}],
                              "options": {"num_ctx": a.num_ctx, "num_predict": 4}})
    ps = post_get("/api/ps")
    m = next((x for x in ps.get("models", []) if x.get("name") == a.model), {})
    meta = {"model": a.model, "num_ctx": a.num_ctx, "think": a.think, "load_s": round(time.time() - t0, 1),
            "ollama_size_gb": round(m.get("size", 0) / 1e9, 2), "runner_footprint_gb": runner_footprint_gb(a.model),
            "version": post_get("/api/version").get("version"), "guard_runner_pids": runner_pids(a.model)}
    print(json.dumps(meta), flush=True)
    if not meta["guard_runner_pids"] and a.num_ctx > LARGE_WINDOW:
        try:
            _post("/api/generate", {"model": a.model, "keep_alive": 0}, timeout=UNLOAD_TIMEOUT_S)
        except Exception:
            pass
        sys.exit("REFUSED no runner process found for " + a.model + ": the guard could only ask Ollama to unload")
    with open(path, "a") as out:
        for c in cases:
            if c["id"] in have:
                continue
            if not guard_alive():
                sys.exit("ABORT the memory guard stopped answering; the run does not go on unwatched")
            if v3 is not None:
                row = score_v3_row(v3, c, a.model)
                if row["status"] == "skipped":
                    time.sleep(GUARD_EVERY_S + 1)
                hold_if_aborting()
                out.write(json.dumps(row) + "\n")
                out.flush()
                continue
            req = request(c, a.model)
            if "not_sent" in req:   # Grux 3.3.1 answers this message itself: nothing goes to a model
                out.write(json.dumps({"case": c["id"], "family": c["family"], "status": "not_sent",
                                      "why": req["not_sent"]}) + "\n")
                out.flush()
                continue
            t = time.time()
            try:
                raw = post(req["path"], req["body"])
                ok, why, got = score(c, raw, req["offered"]) if a.set == "holdout2" and "offered" in req \
                    else score(c, raw, req) if a.set == "local" else score(c, raw)
                row = {"case": c["id"], "family": c["family"], "status": "scored", "pass": ok, "why": why,
                       "got": got, "s": round(time.time() - t, 2),
                       "first_token_s": round((raw.get("prompt_eval_duration") or 0) / 1e9 + (raw.get("load_duration") or 0) / 1e9, 2),
                       "out_tokens": raw.get("eval_count")}
                if "offered" in req:
                    row["offered"] = req["offered"]
                    row["prompt_tokens"] = raw.get("prompt_eval_count")
                if a.set == "local":
                    row["tool_list"] = "stable" if req["stable"] else "ranked: " + req["why_ranked"]
            except Exception as e:
                row = {"case": c["id"], "family": c["family"], "status": "skipped", "error": str(e)[:200],
                       "s": round(time.time() - t, 2)}
                time.sleep(GUARD_EVERY_S + 1)   # a runner that died of memory: let the guard see the breach first
            hold_if_aborting()
            out.write(json.dumps(row) + "\n")
            out.flush()
    meta["peak_runner_footprint_gb"] = PEAK["gb"]   # sampled every guard tick; rss_gb keys before this measured the wrong thing
    meta["wall_s"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(a.run, stem + suffix + ".meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


def score_v3_row(v3, c, model):
    """One held-out v3 job, step by step (holdout_v3.run_case), as one row: pass, why, and each step's seconds and
    prompt tokens, so a prompt Ollama trimmed shows as a prompt_tokens number below the request's size."""
    t = time.time()
    try:
        ok, why, got, steps = v3.run_case(c, model, post)
    except Exception as e:
        return {"case": c["id"], "family": c["family"], "status": "skipped", "error": str(e)[:200],
                "s": round(time.time() - t, 2)}
    return {"case": c["id"], "family": c["family"], "status": "scored", "pass": ok, "why": why, "got": got,
            "s": round(time.time() - t, 2), "steps": steps, "prompt_tokens": [x["prompt_tokens"] for x in steps],
            "out_tokens": sum(x["out_tokens"] or 0 for x in steps)}


def post_get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read().decode() or "{}")


if __name__ == "__main__":
    main()
