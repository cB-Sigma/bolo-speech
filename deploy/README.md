# Deployment examples

| File | What it is | Status |
| --- | --- | --- |
| `Dockerfile` | CPU image (Python 3.12 slim, sherpa-onnx + faster-whisper + Moonshine + PyAV + Wyoming + uccharan; 994 MB) | **built and run** by the maintainers (2026-11-13, Docker 29.9): `/healthz`, the container health check, synthesis with `piper-en-us-amy-low` (uccharan phonemes) and transcription of that audio back with `faster-whisper-tiny`; built and smoke-tested by CI on every push (`deploy/smoke_test.sh`), and before the release workflow pushes it |
| `smoke_test.sh` | starts an image, waits for `/healthz`, synthesises a sentence and transcribes it back | used by CI and the release workflow; run here |
| `Dockerfile.gpu` | NVIDIA CUDA 12 image | **untested** - not built; GPUs were run only on one Windows laptop, natively and under WSL2 ([docs/status.md](../docs/status.md#a-windows-11-laptop)) |
| `docker-compose.yml` | one service, a named volume for models | **run** with the image above: started, reported `healthy`, answered `/healthz` |
| `systemd/bolo.service` | hardened unit for a virtualenv install | `systemd-analyze verify` passes (apart from the missing virtualenv); not run under systemd (no systemd in the build machine) |
| Windows (below) | Task Scheduler, NSSM and WinSW recipes | **not run**; CI runs the tests and `bolo serve` on `windows-latest` |
| `kubernetes/bolo.yaml` | PVC + Deployment + Service with probes | example; not applied to a cluster (none available) |

Building behind a proxy or with a registry mirror: `--build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.12-slim`
replaces the base image. Wheels placed in `deploy/wheels/` are installed in preference to the package index (for a
package that is not published yet, or an offline build). uccharan 0.6 comes from PyPI; build with
`--build-arg EXTRAS=sherpa,faster-whisper,moonshine,av,wyoming` to leave it out (Piper voices then need eSpeak NG).

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

uccharan (Apache-2.0) is in the CPU image: Piper, Kitten and Kokoro voices in its 39 languages work as they are; a
language's word table is downloaded from uccharan's release the first time a voice needs it (set
`BOLO_UCCHARAN_AUTO_DOWNLOAD=false` to refuse, and mount packs fetched with `python -m uccharan.packs --dir` at a path
named in `UCCHARAN_DATA_DIR` for an offline host).
eSpeak NG (GPL-3.0-or-later) is not in the images by default; voices in other languages need it. Build with
`--build-arg WITH_ESPEAK=1` to include it (you then have to meet the GPL's
terms for that component if you pass the image on), or install it into a
running container from the console or with
`bolo components install espeak-ng --accept-license` (the container's Python
environment must be writable; the change is lost when the container is
replaced). See `docs/models.md` for which voices need it.

## The console on its own port

By default one port serves the API and the web console. To expose only the API (a firewall rule, a port mapping, a
proxy) and keep the console private, give the console a port of its own:

```bash
bolo serve --host 0.0.0.0 --port 8000 --console-port 8001 --console-host 127.0.0.1 --api-key "$KEY"
# or BOLO_CONSOLE_PORT=8001 BOLO_CONSOLE_HOST=127.0.0.1, or console_port = 8001 in bolo.toml
```

Port 8000 then answers the API only (`/` is a 404 there); port 8001 serves only the console (its `/api/...` paths
are 404s that point at the API). The console page is told to call the API at its own host name on port 8000, and
the API accepts cross-origin requests from the console's origin (any host name at port 8001; only `localhost`,
`127.0.0.1` and `[::1]` when the console listens on loopback), so nothing else needs configuring. Behind a proxy or
TLS, tell the console where the API is with `--console-api-url https://speech.example.com` (`BOLO_CONSOLE_API_URL`);
if the console is then served from an origin other than port 8001, add it with `--cors-origin`. The console still
needs the API key to do anything (it asks for it, and keeps it in the browser). `--console-port 0` or `--no-console`
serves no console at all. In Docker, publish only the port you want reachable (`-p 8000:8000`).

For voice agents (`/v1/realtime` conversations, audio in `/v1/chat/completions`)
point bolo at a language model: `BOLO_LLM_BASE_URL=http://ollama:11434/v1` and
`BOLO_LLM_MODEL=llama3.2`. See `docs/voice-agents.md`.

## Windows: at log-on, at start-up or as a service

bolo has no Windows installer and needs none: `uv tool install "bolo-speech[all]"` puts `bolo.exe` in
`%USERPROFILE%\.local\bin` (`(Get-Command bolo).Source` prints where). None of the recipes below was run by the
maintainers; CI runs bolo's tests and `bolo serve` on `windows-latest`, nothing more.

Logs: a background bolo has no console, so give it a file: set `BOLO_LOG_FILE=auto` (bolo cannot tell it runs as a
Windows service). The file is `%LOCALAPPDATA%\bolo\logs\server.log` of the account it runs as (for `LocalSystem`,
`C:\Windows\System32\config\systemprofile\AppData\Local\bolo\logs\server.log`); `bolo logs -f` follows it, and the
rotation (10 MB, 3 old files) works while it is open. Models are in `%USERPROFILE%\.cache\bolo` of that account
unless `BOLO_DATA_DIR` says otherwise ([docs/configuration.md](../docs/configuration.md#windows)).

**Task Scheduler** (no extra software; runs as you, with your models and settings). In PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("BOLO_LOG_FILE", "auto", "User")   # read by tasks started after this
$action   = New-ScheduledTaskAction -Execute (Get-Command bolo).Source -Argument "serve"
$trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME        # or: -AtStartup (then see below)
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "bolo" -Action $action -Trigger $trigger -Settings $settings `
  -Description "bolo speech server"
Start-ScheduledTask -TaskName "bolo"          # now, without logging off and on
```

`-ExecutionTimeLimit ([TimeSpan]::Zero)` keeps Task Scheduler from stopping it after three days. `-RestartCount`
and `-RestartInterval` only apply when the task fails to start: once bolo is running, a crash is recorded as the
task's "Last Run Result" and nothing starts it again. To restart bolo after a crash, have the task run a small
restart loop instead ([At log-on, restarted after a crash](../docs/configuration.md#at-log-on-restarted-after-a-crash)),
or use a service manager below. To run it at start-up before anyone logs on,
use `-AtStartup` and register it with your account and password (`-User $env:USERNAME -Password ...`, "run whether
user is logged on or not"), which needs an administrator PowerShell. `Stop-ScheduledTask bolo` stops it (the
servers bolo started stop with it), `Unregister-ScheduledTask bolo` removes it.

**A real service, with [NSSM](https://nssm.cc/)** (administrator PowerShell; the service runs as `LocalSystem`
unless you set `ObjectName`):

```powershell
nssm install bolo (Get-Command bolo).Source serve
nssm set bolo AppEnvironmentExtra BOLO_LOG_FILE=auto BOLO_DATA_DIR=C:\ProgramData\bolo
nssm set bolo AppExit Default Restart
nssm set bolo AppStopMethodSkip 6        # stop with Ctrl+C's console event, as a terminal would
nssm start bolo
```

**Or with [WinSW](https://github.com/winsw/winsw)**: put `WinSW-x64.exe` next to this `bolo-service.xml`, rename
both to `bolo-service.exe` / `bolo-service.xml`, then `.\bolo-service.exe install` and `.\bolo-service.exe start`:

```xml
<service>
  <id>bolo</id>
  <name>bolo speech server</name>
  <executable>C:\Users\you\.local\bin\bolo.exe</executable>
  <arguments>serve</arguments>
  <env name="BOLO_LOG_FILE" value="auto"/>
  <env name="BOLO_DATA_DIR" value="C:\ProgramData\bolo"/>
  <onfailure action="restart" delay="10 sec"/>
  <log mode="roll-by-size"><sizeThreshold>10240</sizeThreshold><keepFiles>3</keepFiles></log>
</service>
```

A service that runs as `LocalSystem` does not see your user's `.cache\bolo`: point `BOLO_DATA_DIR` at a folder it
can write (as above) and pull the models again there, or run it as your account. Each child server bolo starts
(llama-server, audio.cpp, plugin workers) is in a Job Object, so stopping the service or the task stops them too.

**Firewall.** `bolo serve` listens on `127.0.0.1` by default and Windows asks nothing. With `--host 0.0.0.0`,
Windows Defender Firewall asks the first time (or silently blocks it in a service). The prompt names `python.exe`
("Python"), not `bolo.exe`: with uv, `bolo.exe` is a small launcher that starts uv's Python interpreter, and that
is the program that listens. A rule made from the prompt is for that interpreter's path, which changes when uv
installs another Python, so allow the port explicitly instead, and set `BOLO_API_KEY` before you do:

```powershell
New-NetFirewallRule -DisplayName "bolo" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```
