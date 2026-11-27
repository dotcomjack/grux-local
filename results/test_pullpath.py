#!/usr/bin/env python3
"""check.pull_path against a local registry that answers the way gruxai.com does, and the ways it has failed. No
network beyond 127.0.0.1.

  1. HEAD answered on the registry host, GET 307 to another host that answers 206: PASS;
  2. HEAD 307 to another host (2026-11-24, every :2b pull failed "blocked redirect to a different host"): FAIL;
  3. GET 200 with no Location (a preview host; Ollama: "http: no Location header in response"): FAIL;
  4. the Location answers 404: FAIL;
  5. HEAD's length differs from the manifest's: FAIL;
  6. a tag with tensor layers (MLX, the transfer path) is skipped, not judged.
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
            body = json.dumps(MLX if path.endswith("/mlx") else GGUF).encode()
            return self._send(200, [("Content-Length", str(len(body)))], body)
        if "/blobs/" not in path:
            return self._send(404, [("Content-Length", "0")])
        on_registry = host.startswith("127.0.0.1")
        other = f"http://localhost:{PORT}{path}"
        if not on_registry:   # the blob host
            if m == "bad_target":
                return self._send(404, [("Content-Length", "0")])
            return self._send(206, [("Content-Length", "1"), ("Content-Range", "bytes 0-0/10")], BLOB[:1])
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
srv.shutdown()
if not results:
    print("FAIL no check ran")
    sys.exit(1)
sys.exit(0 if all(results) else 1)
