# Network watch: where Ollama connects during a Grux Local chat (2026-10-23)

Who and where: Grux, on a 16 GB Mac mini (Apple M4) with Ollama 0.40.8, on October 23, 2026, at 1:27 PM
Eastern. One full Grux chat turn (Grux's own system prompt and 12 tools, "add a task to call the dentist
tomorrow") sent to `grux-local:2b` (manifest `sha256:601cd23c66ef`, the same build as
`gruxai.com/grux/grux-local:2b`) on the Mac's own Ollama, while `lsof -nP -a -c ollama -i` listed every network
socket held by an Ollama process, five times a second, from before the request until after the reply.

The reply called `add_task` with `{"priority": "next", "title": "Call the dentist tomorrow"}`.

Every distinct socket seen during the chat, verbatim:

```
TCP 127.0.0.1:11434 (LISTEN)
TCP 127.0.0.1:11434->127.0.0.1:49973 (ESTABLISHED)
TCP 127.0.0.1:11434->127.0.0.1:49996 (ESTABLISHED)
TCP 127.0.0.1:49974 (LISTEN)
TCP 127.0.0.1:49974->127.0.0.1:49997 (ESTABLISHED)
TCP 127.0.0.1:49997->127.0.0.1:49974 (ESTABLISHED)
```

All six are on 127.0.0.1: the Ollama server, the request, and the server's own connection to its model runner.
None reaches another host. A sample every 200 ms can miss a connection shorter than that; the model download is a
separate, one-time step and is not part of a chat.
