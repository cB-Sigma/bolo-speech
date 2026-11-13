#!/usr/bin/env bash
# Every HTTP endpoint with curl. Usage: examples/curl.sh [audio-file]
set -euo pipefail
URL="${BOLO_URL:-http://127.0.0.1:8000}"
AUTH=()
[ -n "${BOLO_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer ${BOLO_API_KEY}")
c() { curl -fsS "${AUTH[@]}" "$@"; }

echo "# server";             c "$URL/api/v1/info"; echo
echo "# installed models";   c "$URL/v1/models"; echo

echo "# speech -> hello.wav"
c "$URL/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"alloy","input":"Hello from bolo.","response_format":"wav"}' -o hello.wav

AUDIO="${1:-hello.wav}"
echo "# transcription (json)"
c "$URL/v1/audio/transcriptions" -F "file=@${AUDIO}" -F model=whisper-1; echo
echo "# transcription (verbose_json with word timestamps)"
c "$URL/v1/audio/transcriptions" -F "file=@${AUDIO}" -F model=whisper-1 \
  -F response_format=verbose_json -F 'timestamp_granularities[]=word' -F 'timestamp_granularities[]=segment'; echo
echo "# transcription (server-sent events)"
c -N "$URL/v1/audio/transcriptions" -F "file=@${AUDIO}" -F model=whisper-1 -F stream=true; echo

echo "# telephony: 8 kHz mu-law out, raw mu-law in"
c "$URL/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"input":"This is a phone call.","response_format":"mulaw"}' -o call.ulaw
c "$URL/v1/audio/transcriptions" -F "file=@call.ulaw;type=audio/basic" -F model=whisper-1; echo

echo "# who spoke when, and a transcript with speaker labels (downloads the diarization models on first use)"
c "$URL/api/v1/diarize" -F "file=@${AUDIO}"; echo
c "$URL/v1/audio/transcriptions" -F "file=@${AUDIO}" -F model=whisper-1 -F diarize=true -F response_format=diarized_json; echo

echo "# optional components (eSpeak NG is not shipped; installing it means accepting the GPL-3.0-or-later licence)"
c "$URL/api/v1/components"; echo
# c "$URL/api/v1/components/espeak-ng/install" -H 'Content-Type: application/json' -d '{"accept_license":true}'

echo "# voice activity timestamps (ms)"
c "$URL/v1/audio/speech/timestamps" -F "file=@${AUDIO}"; echo

echo "# catalog search, pull, job, load, stats, traces"
c "$URL/api/v1/catalog?task=stt&q=hindi" | head -c 400; echo
JOB=$(c "$URL/api/v1/models/pull" -H 'Content-Type: application/json' -d '{"id":"silero-vad"}' | sed 's/.*"job_id":"\([^"]*\)".*/\1/')
c "$URL/api/v1/jobs/${JOB}"; echo
c "$URL/api/v1/stats" | head -c 400; echo
c "$URL/api/v1/requests?limit=2" | head -c 600; echo
echo "# prometheus"; c "$URL/metrics" | head -n 8
