# Deployment examples

| File | What it is | Status |
| --- | --- | --- |
| `Dockerfile` | CPU image (Python 3.12 slim, sherpa-onnx + faster-whisper + PyAV + Wyoming) | built by release CI; not built locally by the maintainers for 0.1.0 |
| `Dockerfile.gpu` | NVIDIA CUDA 12 image | **untested** - no GPU was available during development |
| `docker-compose.yml` | one service, a named volume for models | follows the CPU image |
| `systemd/bolo.service` | hardened unit for a virtualenv install | reviewed, not run on a host by the maintainers |
| `kubernetes/bolo.yaml` | PVC + Deployment + Service with probes | example; not applied to a cluster |

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

eSpeak NG (GPL-3.0-or-later) is not in the images by default. Build with
`--build-arg WITH_ESPEAK=1` to include it (you then have to meet the GPL's
terms for that component if you pass the image on), or install it into a
running container from the console or with
`bolo components install espeak-ng --accept-license` (the container's Python
environment must be writable; the change is lost when the container is
replaced). See `docs/models.md` for which voices need it.

For voice agents (`/v1/realtime` conversations, audio in `/v1/chat/completions`)
point bolo at a language model: `BOLO_LLM_BASE_URL=http://ollama:11434/v1` and
`BOLO_LLM_MODEL=llama3.2`. See `docs/voice-agents.md`.
