#!/usr/bin/env python3
"""check.pull_path against a local registry that answers the way gruxai.com does, and the ways it has failed. No
network beyond 127.0.0.1.

  1. HEAD answered on the registry host, GET 307 to another host that answers 206: PASS;
  2. HEAD 307 to another host (2026-11-24, every :2b pull failed "blocked redirect to a different host"): FAIL;
  3. GET 200 with no Location (a preview host; Ollama: "http: no Location header in response"): FAIL;
  4. the Location answers 404: FAIL;
  5. HEAD's length differs from the manifest's: FAIL;
  6. a tag with tensor layers (MLX, the transfer path) is skipped, not judged;
  7. the config blob is walked too (Ollama pulls it): a config that answers no Location fails;
  8. a Location that drops https to http fails on an https registry (checked on the rule, no TLS server);
  9. a 404 carrying a Location fails (Ollama accepts only 200 or a redirect);
 10. a manifest with no layers is one FAIL line, not a traceback that stops the run;
 11. a relative Location is resolved against the blob URL, as Go does;
 12. a layer with no digest is a problem on that tag, not a traceback.
Prints PASS or FAIL per check; exit 1 on any FAIL, or if no check ran.

  python3 results/test_pullpath.py
"""
import contextlib
import http.server
import io
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check  # noqa: E402

BLOB = b"0123456789"
D = "sha256:" + "a" * 64
CFG = "sha256:" + "c" * 64
GGUF = {"schemaVersion": 2, "config": {"digest": CFG, "size": 10, "mediaType": "application/vnd.docker.container.image.v1+json"},
        "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": D, "size": 10}]}
MLX = {"schemaVersion": 2, "config": {"digest": CFG, "size": 10, "mediaType": "application/vnd.docker.container.image.v1+json"},
       "layers": [{"mediaType": "application/vnd.ollama.image.tensor", "digest": D, "size": 10}]}
MODE = {"m": "good"}
results = []


def check_(name, ok):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name)


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, headers=(), body=b""):
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        if body and self.command == "GET":
            self.wfile.write(body)

    def handle_one(self):
        host = self.headers.get("Host", "")
        m, path = MODE["m"], self.path
        if "/manifests/" in path:
            body = json.dumps(MLX if path.endswith("/mlx") else {"schemaVersion": 2} if path.endswith("/broken") else {**GGUF, "layers": GGUF["layers"] + [{"mediaType": "application/vnd.ollama.image.params", "size": 1}]} if path.endswith("/nodigest") else GGUF).encode()
            return self._send(200, [("Content-Length", str(len(body)))], body)
        if "/blobs/" not in path:
            return self._send(404, [("Content-Length", "0")])
        on_registry = host.startswith("127.0.0.1") and "?r=1" not in path
        other = f"http://localhost:{PORT}{path}"
        if not on_registry:   # the blob host
            if m == "bad_target":
                return self._send(404, [("Content-Length", "0")])
            return self._send(206, [("Content-Length", "1"), ("Content-Range", "bytes 0-0/10")], BLOB[:1])
        if m == "config_no_loc" and CFG[7:] in path and self.command == "GET":
            return self._send(200, [("Content-Length", "10")], BLOB)
        if m == "status_404" and self.command == "GET":
            return self._send(404, [("Location", other), ("Content-Length", "0")])
        if m == "relative" and self.command == "GET":
            return self._send(307, [("Location", path + "?r=1"), ("Content-Length", "0")])
        if self.command == "HEAD":
            if m == "head_cross":
                return self._send(307, [("Location", other), ("Content-Length", "0")])
            return self._send(200, [("Content-Length", "9" if m == "bad_len" else "10")])
        if m == "no_location":
            return self._send(200, [("Content-Length", "10")], BLOB)
        return self._send(307, [("Location", other), ("Content-Length", "0")])

    do_GET = do_HEAD = handle_one


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
REG = f"http://127.0.0.1:{PORT}"


def run(mode, tags=("2b",)):
    MODE["m"] = mode
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        failed = check.pull_path(REG, {"tags": {t: {} for t in tags}})
    return failed, out.getvalue()


f, o = run("good")
check_("1. HEAD on the registry, GET 307 to a blob host that answers 206: PASS", f == 0 and "PASS pullpath 2b" in o)
f, o = run("head_cross")
check_("2. a cross-host HEAD redirect fails (the 2026-11-24 outage)", f == 1 and "cross-host" in o)
f, o = run("no_location")
check_("3. a GET with no Location fails (a preview host)", f == 1 and "no Location" in o)
f, o = run("bad_target")
check_("4. a Location that does not answer 206 fails", f == 1 and "not 206" in o)
f, o = run("bad_len")
check_("5. a HEAD length that differs from the manifest fails", f == 1 and "want 10" in o)
f, o = run("head_cross", tags=("mlx",))
check_("6. a tensor (MLX) tag is skipped, not judged", f == 0 and o == "")
f, o = run("config_no_loc")
check_("7. the config blob is walked: its missing Location fails", f == 1 and CFG[7:19] in o and "no Location" in o)
# 8: an https registry cannot be served here, so both network calls are replaced for this one case.
real_raw, real_fetch = check._raw, check.fetch
check.fetch = lambda url, headers=None: (json.dumps(GGUF).encode(), {})
check._raw = lambda method, url, headers=None: ((200, {"content-length": "10"}) if method == "HEAD" else
                                               (206, {}) if headers else (307, {"location": "http://blobs.example/x"}))
out = io.StringIO()
with contextlib.redirect_stdout(out):
    f8 = check.pull_path("https://registry.example", {"tags": {"2b": {}}})
check._raw, check.fetch = real_raw, real_fetch
check_("8. a Location that drops https to http fails", f8 == 1 and "drops https" in out.getvalue())
f, o = run("status_404")
check_("9. a 404 carrying a Location fails", f == 1 and "accepts only 200 or a redirect" in o)
f, o = run("good", tags=("broken",))
check_("10. a manifest with no layers is one FAIL line", f == 1 and "no usable manifest" in o)
f, o = run("relative")
check_("11. a relative Location is resolved against the blob URL", f == 0 and "PASS pullpath" in o)
f, o = run("good", tags=("nodigest",))
check_("12. a layer with no digest is a problem on that tag, not a traceback", f == 1 and "FAIL pullpath nodigest" in o)
srv.shutdown()
if not results:
    print("FAIL no check ran")
    sys.exit(1)
sys.exit(0 if all(results) else 1)
