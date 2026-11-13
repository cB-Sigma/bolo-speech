# bolo, CPU image. Build from the repository root:
#   docker build -f deploy/Dockerfile -t bolo:cpu .
#
# Built and run by the maintainers (2026-11-13, see deploy/README.md) and by CI on every push (smoke test).
# PYTHON_IMAGE lets you use a mirror, e.g. --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.12-slim
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE} AS build
WORKDIR /src
COPY server/ ./server/
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist server

FROM ${PYTHON_IMAGE}
ARG EXTRAS="sherpa,faster-whisper,moonshine,av,wyoming,uccharan"
# libgomp: OpenMP runtime used by CTranslate2 (faster-whisper).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# uccharan (Apache-2.0) is the phonemiser for Piper, Kitten, Matcha and Kokoro voices in English and Hindi.
# eSpeak NG (GPL-3.0-or-later) is NOT in the image by default; phoneme-based voices in other languages need it.
# Character- and lexicon-based voices (MMS-TTS, SYSPIN, MeloTTS, Kokoro Mandarin) need neither.
# Add it at build time with:   docker build --build-arg WITH_ESPEAK=1 ...
# (or at run time from the console / `bolo components install espeak-ng --accept-license`).
# If you redistribute an image built with WITH_ESPEAK=1 you must meet the GPL's terms for that component.
ARG WITH_ESPEAK=0
COPY --from=build /dist/*.whl /tmp/
# Wheels placed in deploy/wheels/ are preferred over the index (for packages not published yet, or offline builds).
COPY deploy/wheels/ /tmp/wheels/
RUN WHEEL="$(ls /tmp/*.whl)" \
    && if [ "${WITH_ESPEAK}" = "1" ]; then EXTRAS="${EXTRAS},espeak"; fi \
    && pip install --no-cache-dir --find-links /tmp/wheels "${WHEEL}[${EXTRAS}]" \
    && rm -rf /tmp/*.whl /tmp/wheels \
    && useradd --create-home --uid 10001 bolo \
    && mkdir -p /data && chown bolo:bolo /data
USER bolo
ENV BOLO_DATA_DIR=/data \
    BOLO_HOST=0.0.0.0 \
    BOLO_PORT=8000
VOLUME ["/data"]
EXPOSE 8000 10300
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"
ENTRYPOINT ["bolo"]
CMD ["serve"]
