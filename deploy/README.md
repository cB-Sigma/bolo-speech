# Deployment examples

| File | What it is | Status |
| --- | --- | --- |
| `Dockerfile` | CPU image (Python 3.12 slim, sherpa-onnx + faster-whisper + Moonshine + PyAV + Wyoming + uccharan; 994 MB) | **built and run** by the maintainers (2026-11-13, Docker 29.9): `/healthz`, the container health check, synthesis with `piper-en-us-amy-low` (uccharan phonemes) and transcription of that audio back with `faster-whisper-tiny`; built and smoke-tested by CI on every push (`deploy/smoke_test.sh`), and before the release workflow pushes it |
| `smoke_test.sh` | starts an image, waits for `/healthz`, synthesises a sentence and transcribes it back | used by CI and the release workflow; run here |
| `Dockerfile.gpu` | NVIDIA CUDA 12 image | **untested** - no GPU was available during development |
| `docker-compose.yml` | one service, a named volume for models | **run** with the image above: started, reported `healthy`, answered `/healthz` |
| `systemd/bolo.service` | hardened unit for a virtualenv install | `systemd-analyze verify` passes (apart from the missing virtualenv); not run under systemd (no systemd in the build machine) |
| `kubernetes/bolo.yaml` | PVC + Deployment + Service with probes | example; not applied to a cluster (none available) |

Building behind a proxy or with a registry mirror: `--build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.12-slim`
replaces the base image. Wheels placed in `deploy/wheels/` are installed in preference to the package index (for a
package that is not published yet, or an offline build). Until uccharan is on PyPI, put its wheel there
(`maturin build --release` in uccharan's `bindings/python`), or build with
`--build-arg EXTRAS=sherpa,faster-whisper,moonshine,av,wyoming` (English and Hindi Piper voices then need eSpeak NG).

All of them keep models in a volume (`BOLO_DATA_DIR`) and bind the HTTP port.
Set `BOLO_API_KEY` before exposing the port to anything but localhost, and
terminate TLS in a reverse proxy. For WebSockets through nginx:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_buffering off;            # streamed audio and Server-Sent Events
    proxy_read_timeout 3600s;
    client_max_body_size 200m;
}
```

uccharan (Apache-2.0) is in the CPU image: English and Hindi Piper, Kitten and Kokoro voices work as they are.
eSpeak NG (GPL-3.0-or-later) is not in the images by default; voices in other languages need it. Build with
`--build-arg WITH_ESPEAK=1` to include it (you then have to meet the GPL's
terms for that component if you pass the image on), or install it into a
running container from the console or with
`bolo components install espeak-ng --accept-license` (the container's Python
environment must be writable; the change is lost when the container is
replaced). See `docs/models.md` for which voices need it.

For voice agents (`/v1/realtime` conversations, audio in `/v1/chat/completions`)
point bolo at a language model: `BOLO_LLM_BASE_URL=http://ollama:11434/v1` and
`BOLO_LLM_MODEL=llama3.2`. See `docs/voice-agents.md`.
