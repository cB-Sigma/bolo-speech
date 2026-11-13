#!/usr/bin/env bash
# Smoke test of a bolo container image: start it, wait for /healthz, synthesise a sentence and transcribe it back.
# Used by CI on every push and by the release workflow before the image is pushed.
#   deploy/smoke_test.sh bolo:cpu [port]
# Downloads two small models into a temporary volume (about 140 MB). Extra `docker run` arguments can be given
# in SMOKE_DOCKER_ARGS (for example a proxy).
set -euo pipefail
IMAGE="${1:?usage: smoke_test.sh <image> [port]}"
PORT="${2:-8077}"
NAME="bolo-smoke-$$"
DATA="$(mktemp -d)"
chmod 777 "$DATA"
cleanup() {
  docker logs "$NAME" > "$DATA/container.log" 2>&1 || true
  docker rm -f "$NAME" > /dev/null 2>&1 || true
  rm -rf "$DATA"
}
trap cleanup EXIT

# The default voice uses uccharan when the image has it; otherwise a character voice that needs no phonemiser.
if docker run --rm --entrypoint python "$IMAGE" -c "import uccharan" > /dev/null 2>&1; then
  VOICE=piper-en-us-amy-low
else
  VOICE=mms-tts-eng
fi
# shellcheck disable=SC2086
docker run -d --name "$NAME" --network host -v "$DATA:/data" \
  -e BOLO_PORT="$PORT" -e BOLO_HOST=127.0.0.1 -e BOLO_DEFAULT_STT=faster-whisper-tiny -e BOLO_DEFAULT_TTS="$VOICE" \
  ${SMOKE_DOCKER_ARGS:-} "$IMAGE" > /dev/null
URL="http://127.0.0.1:$PORT"
for _ in $(seq 1 60); do
  curl -fsS "$URL/healthz" > /dev/null 2>&1 && break
  sleep 1
done
curl -fsS "$URL/healthz"
echo
curl -fsS "$URL/api/v1/info" | python3 -c "import json,sys; d=json.load(sys.stdin); print('bolo', d['version'], [e['id'] for e in d['engines'] if e['available']])"
curl -fsS -o "$DATA/speech.wav" "$URL/v1/audio/speech" -H 'content-type: application/json' \
  -d "{\"model\": \"$VOICE\", \"input\": \"Testing one two three.\", \"response_format\": \"wav\"}"
test "$(stat -c %s "$DATA/speech.wav")" -gt 10000
TEXT="$(curl -fsS "$URL/v1/audio/transcriptions" -F file=@"$DATA/speech.wav" -F model=faster-whisper-tiny -F response_format=text)"
echo "round trip ($VOICE -> faster-whisper-tiny): $TEXT"
echo "$TEXT" | tr '[:upper:]' '[:lower:]' | grep -Eq "test|three|3" || { echo "unexpected transcript" >&2; exit 1; }
echo "smoke test passed"
