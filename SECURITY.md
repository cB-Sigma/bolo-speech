# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's
[private vulnerability reporting](https://github.com/cB-Sigma/bolo-speech/security/advisories/new)
instead. You should get an acknowledgement within a week. Please include the
version (`bolo --version`), how to reproduce the problem, and what an attacker
gains.

Only the latest release is supported with security fixes while the project is
at 0.x.

## What to know before exposing bolo to a network

bolo is a server that runs model files and decodes untrusted audio, so treat
it like any other service that parses complex input.

- **It binds to 127.0.0.1 by default and has no authentication by default.**
  If you bind another address, set an API key (`--api-key` / `BOLO_API_KEY`)
  and put TLS in front of it (a reverse proxy). The key is a single shared
  bearer token: there are no users or scopes.
- **`/healthz`, `/readyz`, `/metrics` and the web console's static files are
  never authenticated.** `/metrics` reveals model names and traffic volumes.
  Keep it off the public internet or filter it at the proxy.
- **Model management is part of the API.** Anyone who can call the API can
  make the server download any model in the catalog, and with
  `POST /api/v1/models/pull {"id": "hf:<repo>"}` any Hugging Face repository
  whose layout bolo recognises. Models are data files run by ONNX Runtime or
  CTranslate2 (no pickles, no Python code is executed from a model
  repository), but they are still large untrusted inputs to native code.
  Start the server with `--no-auto-pull` and pre-install models if clients
  are not trusted.
- **Installing optional components runs a package installer.**
  `POST /api/v1/components/espeak-ng/install` makes the server run
  `pip install espeakng-loader` (or `uv pip install`) in its own environment,
  with a fixed package name: the request cannot choose what is installed.
  It is allowed only when the server is bound to a loopback address or an
  API key is set, and is always refused on a non-loopback bind without a
  key. Set `allow_component_install = false` to switch it off entirely.
- **The language-model connection.** With `llm_base_url` set,
  `/v1/chat/completions` and `/v1/realtime` forward request bodies (and
  transcripts of the audio) to that endpoint with `llm_api_key`, so anyone
  with access to bolo can spend that key's quota. Realtime tool calls are
  only reported to the client as events; bolo never executes a tool itself.
- **Request traces are kept in memory** (the most recent 1,000 by default)
  and include a short preview of transcripts and input text. They are
  visible to anyone with API access through `/api/v1/requests` and
  `/api/v1/events`. Audio is never written to disk.
- **Uploads are limited** to 200 MB by default (`max_upload_mb`). Decoding
  happens in process through libsndfile and, if installed, FFmpeg (PyAV).
- **A new model is first loaded in a child process** so that a malformed
  model file that makes the native runtime abort cannot take the server down
  (`probe_models`, on by default).
- The Wyoming port (off by default) has no authentication at all, as the
  protocol defines none. Bind it to a trusted interface only.
