# bolo, CPU image. Build from the repository root:
#   docker build -f deploy/Dockerfile -t bolo:cpu .
#
# NOTE: this Dockerfile has been written and reviewed but not built by the
# maintainers so far (the release CI builds it). Report problems, please.
FROM python:3.12-slim AS build
WORKDIR /src
COPY server/ ./server/
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist server

FROM python:3.12-slim
ARG EXTRAS="sherpa,faster-whisper,av,wyoming"
# libgomp: OpenMP runtime used by CTranslate2 (faster-whisper).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# eSpeak NG (GPL-3.0-or-later) is NOT in the image by default. Piper-format voices, KittenTTS and the
# non-English Kokoro voices need it; Kokoro English/Mandarin, MeloTTS, MMS-TTS and the Coqui voices do not.
# Add it at build time with:   docker build --build-arg WITH_ESPEAK=1 ...
# (or at run time from the console / `bolo components install espeak-ng --accept-license`).
# If you redistribute an image built with WITH_ESPEAK=1 you must meet the GPL's terms for that component.
ARG WITH_ESPEAK=0
COPY --from=build /dist/*.whl /tmp/
RUN WHEEL="$(ls /tmp/*.whl)" \
    && if [ "${WITH_ESPEAK}" = "1" ]; then EXTRAS="${EXTRAS},espeak"; fi \
    && pip install --no-cache-dir "${WHEEL}[${EXTRAS}]" \
    && rm /tmp/*.whl \
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
