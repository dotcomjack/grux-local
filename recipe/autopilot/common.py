"""Shared pieces of the Grux Local autopilot: paths, the spend meter, the base row per Mac tier, intervals.

Raw run output (the per-task outbox, logs) lives outside the repo under STATE; the repo gets only the summary
JSON per model in results/autopilot/ (progress-saving rule: commit the summary, never the raw output).
"""
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.dirname(HERE)
REPO = os.path.dirname(RECIPE)
# The scheduled job runs from an installed copy of the lane (install_launchd.py) and writes its results to a stable
# home outside any checkout; run by hand from the repo, results land in the repo's results/autopilot/.
RESULTS = os.path.expanduser(os.environ.get("GRUX_LOCAL_AUTOPILOT_RESULTS", os.path.join(REPO, "results", "autopilot")))
GATE_REQUESTS = os.path.join(RESULTS, "gate-requests")
STATE = os.path.expanduser(os.environ.get("GRUX_LOCAL_AUTOPILOT_STATE", "~/.grux/autopilot/grux-local"))
READY = os.path.expanduser(os.environ.get("GRUX_AUTOPILOT_READY", "~/.grux/autopilot/ready"))
HEARTBEAT = os.path.expanduser(os.environ.get("GRUX_AUTOPILOT_HEARTBEAT", "~/.grux/autopilot/heartbeat/grux-local.json"))
SPEND_TOOL = os.path.expanduser(os.environ.get("DAYTONA_SPEND_TOOL", "~/.claude/tools/daytona_spend.py"))
LANE = "grux-local"
OWNER = "grux-local"          # the dcj.owner label on every box this lane makes
SWING_LIMIT_USD = 150.0       # one swing estimated over this stops and asks the owner; never a silent create
# Listed spot rate per hour for the GPU, plus the 16 vCPU and 192 GiB every GPU box also bills ($3.92 an hour,
# the same formula as the meter). Not a settled charge.
BOX_EXTRA = 16 * 0.0504 + 192 * 0.0162
GPU_RATE = {"H100": 2.27, "H200": 2.61, "B200": 3.59, "B300": 4.08, "RTX-PRO-6000": 1.74, "RTX-4090": 0.57,
            "RTX-5090": 0.74}

sys.path[:0] = [RECIPE, os.path.join(RECIPE, "s2")]


def ensure_grux_src():
    """The scorer reads Grux's prompts from the public Grux source at the pinned tag (recipe/get_grux_source.sh)."""
    if not os.environ.get("GRUX_SRC"):
        out = subprocess.run(["sh", os.path.join(RECIPE, "get_grux_source.sh")], capture_output=True, text=True)
        os.environ["GRUX_SRC"] = out.stdout.strip()
    return os.environ["GRUX_SRC"]


def base_config():
    return json.load(open(os.path.join(HERE, "base.json")))


def today():
    return dt.date.today().isoformat()


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def stem(model):
    return model.replace(":", "_").replace("/", "_")


def log(msg, path=None):
    line = dt.datetime.now().strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(line + "\n")


def write_json(path, obj):
    """Atomic: temp then rename, so a crash never leaves half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- Mac tiers

def params_from_tag(tag):
    """Billions of parameters from an Ollama size tag ('4b', 'e4b', '1.7b', '30b-a3b' -> 30). None when absent."""
    m = re.search(r"(?<![a-z0-9.])e?(\d+(?:\.\d+)?)b(?![a-z0-9])", tag.lower())
    return float(m.group(1)) if m else None


def tier_for(params_b):
    """The Mac tier a base of this size competes in, named for the published tag it would replace:
    under 3.5B -> the 8 GB tier (2b), under 6.5B -> 16 GB (4b), up to 10.5B -> 24 GB (9b). None outside 1B to 10.5B."""
    if params_b is None or params_b < 1.0 or params_b > 10.5:
        return None
    return "2b" if params_b < 3.5 else "4b" if params_b < 6.5 else "9b"


def fits_tier(download_bytes, tier):
    """Loaded size estimate (download x 1.4, the ratio the Cookbook's entries carry) inside 75 percent of the tier's
    RAM (Apple silicon's Metal working set). Returns (fits, estimate_gb, budget_gb)."""
    ram = base_config()["tiers"][tier]["ram_gb"]
    est = round(download_bytes * 1.4 / 1e9, 2)
    budget = round(0.75 * ram, 2)
    return est <= budget, est, budget


# --------------------------------------------------------------------------- intervals

def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def newcombe(k1, n1, k2, n2):
    """95% interval of p1 - p2 (Newcombe's hybrid score method, built on the two Wilson intervals). Unpaired, so
    for the same tasks scored twice it is wider than a paired test: a conservative bar for calling a win."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return [round(lo, 4), round(hi, 4)]


def cell(k, n):
    return {"pass": k, "of": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


# --------------------------------------------------------------------------- spend

class Refused(Exception):
    pass


def swing_guard(est_usd, what):
    """Code-level refusal: one swing estimated over SWING_LIMIT_USD never creates a box; the owner decides."""
    if est_usd > SWING_LIMIT_USD:
        raise Refused(f"SWING REFUSED: {what} is estimated at ${est_usd:.2f}, over the ${SWING_LIMIT_USD:.0f} "
                      "single-swing line. Ask the owner before running it. No box was created.")


def box_rate(gpu):
    return GPU_RATE[gpu] + BOX_EXTRA


def reserve(hours, gpu):
    """Reserve on the shared weekly meter BEFORE a create. Exit 3 from the meter is a refusal: raise, create nothing."""
    gpu_key = gpu.replace("-", "")
    r = subprocess.run([sys.executable, SPEND_TOOL, "reserve", "--lane", LANE, "--hours", f"{hours:.3f}",
                        "--gpu", gpu_key], capture_output=True, text=True)
    if r.returncode == 3:
        raise Refused("METER " + (r.stderr.strip() or "REFUSED: over the weekly cap") + ". No box was created.")
    if r.returncode != 0:
        raise Refused(f"meter failed (exit {r.returncode}): {r.stderr.strip()[:300]}. No box was created.")
    return r.stdout.strip()


def settle(rid, hours):
    r = subprocess.run([sys.executable, SPEND_TOOL, "settle", rid, "--hours", f"{hours:.4f}"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"settle failed: {r.stderr.strip()[:200]}"


# --------------------------------------------------------------------------- ready items

def ready_item(item_id, kind, title, what, evidence, ship_command, surfaces):
    """The shared ready-item schema every lane writes (one file per item)."""
    item = {"id": item_id, "lane": LANE, "kind": kind, "title": title, "what": what, "evidence": evidence,
            "ship_command": ship_command, "surfaces": surfaces, "safety_touch": False, "created_at": now_iso()}
    path = os.path.join(READY, item_id + ".json")
    write_json(path, item)
    return path
