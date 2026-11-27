"""Per num_batch: a cold request, then 3 requests that share the ~5.4k-token system prompt and differ in the user turn.
Prints prompt seconds (Ollama's prompt_eval_duration) and the -b/-ub llama-server got."""
import json, os, subprocess, sys, urllib.request
API = os.environ.get("API", "http://127.0.0.1:11501")
SYS = "You are Grux, a Mac assistant. " + " ".join(f"Rule {i}: keep answers short, name the tool you use, never invent a result." for i in range(260))
def call(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
def chat(model, user, nb):
    r = call("/api/chat", {"model": model, "stream": False, "think": False, "keep_alive": "5m",
                           "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                           "options": {"num_ctx": 8192, "num_predict": 8, "temperature": 0, **({"num_batch": nb} if nb else {})}})
    return round(r.get("prompt_eval_duration", 0) / 1e9, 3)
model = sys.argv[1]
for nb in [int(x) for x in sys.argv[2:]]:
    t = [chat(model, u, nb) for u in ["What is on my calendar today?", "Add a task: call the dentist.", "What is 17 times 3?", "Open my notes."]]
    args = subprocess.run(["sh", "-c", "ps -axo args= | grep llama-server | grep -v grep | grep -o -- '-b [0-9]* -ub [0-9]*'"], capture_output=True, text=True).stdout.strip()
    print(f"num_batch {nb or 'default'}: cold {t[0]} s, then {t[1:]} s   [{args}]", flush=True)
    call("/api/generate", {"model": model, "keep_alive": 0})
