"""Orpheus-style text-to-speech without PyTorch: llama.cpp writes the SNAC codes, onnxruntime decodes them.

Orpheus (Canopy Labs) and the models built on it or on its recipe (Veena, Svara, snorTTS) are plain Llama
language models whose vocabulary was extended with SNAC audio codes. ``llama-server`` runs the language model
from a GGUF file; bolo builds the prompt as token ids, streams the generated ids from ``POST /completion``
(``prompt`` as a token array, ``stream: true``, ``return_tokens: true``), groups them into SNAC frames and decodes
them with the SNAC 24 kHz decoder in onnxruntime (``tools/export_snac_onnx.py``), streaming PCM as it goes.

The server is the one the ``tts-server`` engine manages (``bolo.engines.tts_server``): started by bolo on a
loopback port with a random API key and stopped on unload, or run by the operator and named in the setting
``tts_servers`` (``"<model id or llama-server>=<url>"``). This module only adds the Orpheus client.

Token layout (all Orpheus-family models): audio tokens start at 128266; each 85.3 ms frame (2048 samples at
24 kHz) is 7 tokens, the i-th token of a frame offset by ``i * 4096``; they map onto SNAC's three code levels
as 1 + 2 + 4 codes. Prompts (``options.prompt_format``):

* ``veena`` (Maya Research card): ``[START_OF_HUMAN] "<spk_{voice}> {text}" [END_OF_HUMAN] [START_OF_AI]
  [START_OF_SPEECH]``; temperature 0.4, top-p 0.9, repetition penalty 1.05; stops at END_OF_SPEECH or
  END_OF_AI.
* ``svara`` (Kenpath svara-tts-inference, tts_engine/encoder.py): ``[BOS] [START_OF_HUMAN] [AUDIO] "{speaker}:
  {text}" [END_OF_HUMAN] [END_OF_TURN] [START_OF_AI] [START_OF_SPEECH]`` with speakers such as "Hindi (Male)";
  temperature 0.75, top-p 0.9, repetition penalty 1.1.
* ``orpheus`` (canopylabs/orpheus-tts, and snorTTS-Indic): ``[START_OF_HUMAN] [BOS] "{voice}: {text}"
  [END_OF_TURN] [END_OF_HUMAN] [START_OF_AI] [START_OF_SPEECH]``; Orpheus 0.6 / 0.8 / 1.3, snorTTS 0.4 / 0.9 /
  1.05 (catalog options override the defaults). bolo's Svara entries use this layout too (with Svara's sampling
  and ``speaker_field: name``): with Kenpath's layout the Q4_K_M GGUF spoke the speaker label in Bengali.

The speaker in the prompt is the voice's ``speaker`` if it has one, else its ``name`` (``speaker_field: name``,
the default for ``svara``) or its ``id`` (the default otherwise).

Text is tokenised by the server (``POST /tokenize`` with special tokens parsed, as the Hugging Face tokenizer
does), so the ids always match the GGUF's vocabulary. Sampling is sent per request: temperature and top-p
applied in that order after the repetition penalty (``samplers``), top-k and min-p off, the penalty over the
whole sequence (prompt and budget). Generation stops at the format's end-of-speech id (bolo closes the stream,
which cancels the slot), at llama.cpp's own end-of-generation ids, or at a token budget that follows the length
of the text: ``frames_per_char`` (default 2.0) frames of 7 tokens per character plus 12 frames, at most
``max_new_tokens``. (bolo-torch allows 1.3 frames per character, which cuts off ordinary Bengali and Hindi
sentences: speech in these scripts runs at about 1-1.5 frames per character.)

Audio is decoded while the model generates, exactly as bolo's PyTorch plugin does: every ``stream_frames`` new
frames, the new frames are decoded together with up to ``context_frames`` frames of left context (whose audio
is dropped), and one frame is held back as look-ahead until the end.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx
import numpy as np

from bolo.engines.base import FloatArray, LoadOptions, SynthesisExtras, TTSModel, Voice
from bolo.errors import ComponentMissing, EngineUnavailable, InvalidRequest
from bolo.ort_providers import providers_for

if TYPE_CHECKING:
    from bolo.engines.tts_server import ServerProcess, TTSServerEngine

log = logging.getLogger("bolo.orpheus")

KIND = "orpheus-llamacpp"

BOS = 128000
END_OF_TURN = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
SVARA_AUDIO = 156939
AUDIO_OFFSET = 128266
CODEBOOK = 4096
FRAME = 7
SAMPLES_PER_FRAME = 2048
SAMPLE_RATE = 24000

FORMATS: dict[str, dict[str, Any]] = {
    "veena": {"temperature": 0.4, "top_p": 0.9, "repetition_penalty": 1.05, "eos": [END_OF_SPEECH, END_OF_AI]},
    "svara": {"temperature": 0.75, "top_p": 0.9, "repetition_penalty": 1.1, "eos": [END_OF_SPEECH]},
    "orpheus": {"temperature": 0.6, "top_p": 0.8, "repetition_penalty": 1.3, "eos": [END_OF_SPEECH]},
}


# -- prompt and token layout (the same as bolo-torch's models/orpheus.py) -----------------------------------------
def build_prompt(fmt: str, tokenize: Callable[[str], list[int]], text: str, speaker: str) -> list[int]:
    """The prompt token ids for one piece of text. ``tokenize`` encodes text without adding BOS."""
    if fmt == "veena":
        ids = tokenize(f"<spk_{speaker}> {text}")
        return [START_OF_HUMAN, *ids, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
    if fmt == "svara":
        ids = tokenize(f"{speaker}: {text}")
        return [BOS, START_OF_HUMAN, SVARA_AUDIO, *ids, END_OF_HUMAN, END_OF_TURN, START_OF_AI, START_OF_SPEECH]
    if fmt == "orpheus":
        ids = tokenize(f"{speaker}: {text}" if speaker else text)
        return [START_OF_HUMAN, BOS, *ids, END_OF_TURN, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
    raise InvalidRequest(f"unknown prompt_format {fmt!r} ({', '.join(FORMATS)})")


def is_audio_token(token: int) -> bool:
    return AUDIO_OFFSET <= token < AUDIO_OFFSET + FRAME * CODEBOOK


def frames_from_tokens(tokens: list[int]) -> list[list[int]]:
    """Complete 7-code frames (codes 0..4095) from the audio tokens of a generation, in order."""
    codes = [t - AUDIO_OFFSET - (i % FRAME) * CODEBOOK for i, t in enumerate(t for t in tokens if is_audio_token(t))]
    frames = [codes[i : i + FRAME] for i in range(0, len(codes) - len(codes) % FRAME, FRAME)]
    return [f for f in frames if all(0 <= c < CODEBOOK for c in f)]


def snac_levels(frames: list[list[int]]) -> list[list[int]]:
    """De-interleave frames into SNAC's three levels (1, 2 and 4 codes per frame)."""
    levels: list[list[int]] = [[], [], []]
    for f in frames:
        levels[0].append(f[0])
        levels[1].extend((f[1], f[4]))
        levels[2].extend((f[2], f[3], f[5], f[6]))
    return levels


def token_budget(text: str, max_new_tokens: int, frames_per_char: float = 2.0) -> int:
    """Most tokens to generate for ``text``: a stop for a model that never emits its end-of-speech id."""
    return min(max_new_tokens, (int(len(text) * frames_per_char) + 12) * FRAME)


# -- SNAC decoder -----------------------------------------------------------------------------------------------
class OrtSession(Protocol):
    """The part of ``onnxruntime.InferenceSession`` the decoder uses (tests pass a stand-in)."""

    def get_inputs(self) -> list[Any]: ...

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]: ...


class SnacDecoder:
    """SNAC 24 kHz codes -> audio with an ONNX graph: three int64 inputs (levels 0, 1, 2), audio out.

    Works with bolo's export (``codes_0``, ``codes_1``, ``codes_2``) and with others that name the inputs
    differently (``audio_codes.0`` …): inputs are taken in the order the graph declares them.
    """

    def __init__(self, session: OrtSession) -> None:
        self.session = session
        inputs = session.get_inputs()
        if len(inputs) != 3:
            raise EngineUnavailable(f"the SNAC decoder graph has {len(inputs)} inputs, expected 3")
        self.names = [str(i.name) for i in inputs]
        self.dtype = np.int32 if "int32" in str(getattr(inputs[0], "type", "")) else np.int64

    def decode(self, levels: list[list[int]]) -> FloatArray:
        feeds = {name: np.asarray([level], dtype=self.dtype) for name, level in zip(self.names, levels)}
        out = self.session.run(None, feeds)[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)


def open_session(path: Path, threads: int, device: str = "cpu") -> OrtSession:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ComponentMissing(
            "the Orpheus route decodes audio with onnxruntime, which is not installed: pip install onnxruntime "
            '(or pip install "bolo-speech[orpheus]")'
        ) from exc
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, threads)
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    session: OrtSession = ort.InferenceSession(str(path), opts, providers=providers_for(device))
    return session


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_decoder(path: Path, spec: dict[str, Any], data_dir: Path | None, *, download: bool) -> Path:
    """The SNAC decoder file of a model.

    ``options.snac`` names it: ``file`` (where ``bolo pull`` puts it inside the model folder, from the entry's
    ``source.urls``), and ``url`` + ``sha256`` (+ ``size``). When the model is served by an operator-run
    llama-server nothing was pulled, so the decoder is fetched once into ``<data_dir>/components/snac/`` and
    checked against its SHA-256. ``options.snac_path`` (an absolute path) overrides both.
    """
    if spec.get("snac_path"):
        given = Path(str(spec["snac_path"])).expanduser()
        if not given.is_file():
            raise EngineUnavailable(f"the SNAC decoder {given} (option snac_path) does not exist")
        return given
    info = dict(spec.get("snac") or {})
    name = str(info.get("file") or "")
    if not name:
        raise EngineUnavailable("this catalog entry does not name its SNAC decoder (options.snac.file)")
    local = path / name
    if local.is_file():
        return local
    sha = str(info.get("sha256") or "")
    url = str(info.get("url") or "")
    if data_dir is None or not url or not sha:
        raise EngineUnavailable(f"the SNAC decoder {local} is missing; pull the model again")
    cached = Path(data_dir) / "components" / "snac" / sha[:16] / Path(name).name
    if cached.is_file():
        return cached
    if not download:
        raise EngineUnavailable(f"the SNAC decoder {local} is missing; pull the model again")
    log.info("fetching the SNAC decoder from %s", url)
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(60.0, connect=15.0)) as resp:
            if resp.status_code >= 400:
                raise EngineUnavailable(f"could not fetch the SNAC decoder from {url} (HTTP {resp.status_code})")
            with tmp.open("wb") as out:
                for block in resp.iter_bytes():
                    out.write(block)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise EngineUnavailable(f"could not fetch the SNAC decoder from {url}: {exc}") from exc
    if _sha256(tmp) != sha:
        tmp.unlink(missing_ok=True)
        raise EngineUnavailable(f"the SNAC decoder from {url} does not match its SHA-256; not used")
    tmp.replace(cached)
    return cached


# -- the model --------------------------------------------------------------------------------------------------
class OrpheusLlamaCpp(TTSModel):
    """An Orpheus-family voice: prompt and sampling here, the language model in llama-server, SNAC in onnxruntime."""

    accepts_extras = True

    def __init__(
        self,
        name: str,
        spec: dict[str, Any],
        decoder: SnacDecoder,
        *,
        base_url: str | None = None,
        process: ServerProcess | None = None,
        api_key: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        fmt = str(spec.get("prompt_format") or "")
        if fmt not in FORMATS:
            raise EngineUnavailable(f"model '{name}': unknown prompt_format {fmt!r} ({', '.join(FORMATS)})")
        self.name = name
        self.spec = spec
        self.prompt_format = fmt
        self.decoder = decoder
        self.process = process
        self._base_url = (base_url or (process.base_url if process else "")).rstrip("/")
        defaults = FORMATS[fmt]
        self.temperature = float(spec.get("temperature", defaults["temperature"]))
        self.top_p = float(spec.get("top_p", defaults["top_p"]))
        self.repetition_penalty = float(spec.get("repetition_penalty", defaults["repetition_penalty"]))
        self.eos = {int(t) for t in spec.get("eos_tokens") or defaults["eos"]}
        self.seed = int(spec["seed"]) if spec.get("seed") is not None else None
        self.max_chars = int(spec.get("max_chars", 200))
        self.max_new_tokens = int(spec.get("max_new_tokens", 2048))
        self.frames_per_char = float(spec.get("frames_per_char", 2.0))
        self.stream_frames = max(1, int(spec.get("stream_frames", 4)))
        self.context_frames = max(0, int(spec.get("context_frames", 3)))
        self.request_extra = dict(spec.get("request") or {})
        self.sample_rate = SAMPLE_RATE
        api_key = api_key or (process.api_key if process is not None else None)
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            trust_env=process is None,  # never send loopback traffic of a managed server through a proxy
        )
        voices: list[Voice] = []
        self._speakers: dict[str, str] = {}
        field = str(spec.get("speaker_field") or ("name" if fmt == "svara" else "id"))
        for i, v in enumerate(spec.get("voices") or []):
            vid = str(v["id"])
            voices.append(Voice(vid, str(v.get("name", "")), str(v.get("language", "")), str(v.get("gender", "")), i))
            # Veena and Orpheus name the voice by id (<spk_kavya>, "tara"), Svara by display name ("Hindi (Male)",
            # option speaker_field: name); an explicit "speaker" wins (snorTTS: "hindi159")
            default = str(v.get("name") or vid) if field == "name" else vid
            self._speakers[vid.lower()] = str(v.get("speaker", default))
        self.voices = tuple(voices)

    @property
    def base_url(self) -> str:
        return self.process.base_url if self.process is not None else self._base_url

    # -- setup ----------------------------------------------------------------------------------------------
    def connect(self) -> None:
        """Start (managed) or reach (operator-run) the server and check that it answers."""
        if self.process is not None:
            self.process.ensure_started()
        try:
            resp = self._client.get(self.base_url + "/health", timeout=10.0)
        except httpx.HTTPError as exc:
            raise EngineUnavailable(self._unreachable(exc)) from exc
        if resp.status_code == 503:
            raise EngineUnavailable(f"llama-server at {self.base_url} is still loading its model; try again shortly")
        if resp.status_code >= 400:
            raise EngineUnavailable(f"llama-server at {self.base_url} answered /health with {resp.status_code}")

    def warm_up(self) -> None:
        with contextlib.suppress(Exception):
            self.decoder.decode([[0], [0, 0], [0, 0, 0, 0]])

    def close(self) -> None:
        if self.process is not None:
            self.process.stop()
        self._client.close()

    def _unreachable(self, exc: Exception) -> str:
        where = "bolo's llama-server" if self.process else "llama-server"
        if isinstance(exc, httpx.TimeoutException):
            return f"{where} at {self.base_url} did not answer in time ({type(exc).__name__})"
        return f"{where} at {self.base_url} is not reachable ({type(exc).__name__}: {exc})"

    # -- requests ---------------------------------------------------------------------------------------------
    def speaker(self, voice: str | None) -> str:
        if not self._speakers:
            if not voice:
                raise InvalidRequest(f"model '{self.name}' needs a voice", param="voice")
            return voice
        if not voice:
            return next(iter(self._speakers.values()))
        found = self._speakers.get(voice.strip().lower())
        if found is None:
            cand = self.find_voice(voice)
            found = self._speakers.get(cand.id.lower()) if cand is not None else None
        if found is None:
            raise InvalidRequest(f"model '{self.name}' has no voice {voice!r}", param="voice")
        return found

    def tokenize(self, text: str) -> list[int]:
        resp = self._post("/tokenize", {"content": text, "add_special": False, "parse_special": True})
        tokens = resp.json().get("tokens")
        if not isinstance(tokens, list):
            raise EngineUnavailable(f"llama-server at {self.base_url} answered /tokenize without tokens")
        return [int(t["id"]) if isinstance(t, dict) else int(t) for t in tokens]

    def completion_request(self, prompt: list[int], budget: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": prompt,
            "stream": True,
            "return_tokens": True,
            "n_predict": budget,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": 0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": self.repetition_penalty,
            "repeat_last_n": len(prompt) + budget,  # the whole sequence (this build refuses -1)
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "samplers": ["penalties", "temperature", "top_p"],
            "cache_prompt": True,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        body.update(self.request_extra)
        return body

    def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        for attempt in (1, 2):
            try:
                resp = self._client.post(self.base_url + path, json=body)
            except (httpx.RemoteProtocolError, httpx.ConnectError) as exc:
                if self.process is not None:
                    self.process.check()
                if attempt == 1 and (self.process is None or self.process.running):
                    continue
                raise EngineUnavailable(self._unreachable(exc)) from exc
            except httpx.HTTPError as exc:
                raise EngineUnavailable(self._unreachable(exc)) from exc
            if resp.status_code >= 400:
                self._raise_for(resp)
            return resp
        raise AssertionError("unreachable")  # pragma: no cover

    def _raise_for(self, resp: httpx.Response) -> None:
        from bolo.engines.tts_server import _error_message

        message = _error_message(resp)
        where = f"llama-server at {self.base_url}"
        if resp.status_code in (400, 422):
            raise InvalidRequest(f"{where} refused the request for '{self.name}': {message}")
        if resp.status_code in (401, 403):
            raise EngineUnavailable(f"{where} refused bolo's credentials (set tts_server_api_key): {message}")
        raise EngineUnavailable(f"{where} failed ({resp.status_code}): {message}")

    def generate(self, prompt: list[int], budget: int, cancelled: Callable[[], bool]) -> Iterator[int]:
        """The generated token ids as llama-server streams them; stops at an end-of-speech id or cancellation."""
        body = self.completion_request(prompt, budget)
        for attempt in (1, 2):
            started = False
            try:
                with self._client.stream("POST", self.base_url + "/completion", json=body) as resp:
                    if resp.status_code >= 400:
                        resp.read()
                        self._raise_for(resp)
                    for line in resp.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        doc = json.loads(payload)
                        if isinstance(doc, dict) and doc.get("error"):
                            err = doc["error"]
                            message = err.get("message") if isinstance(err, dict) else err
                            raise EngineUnavailable(f"llama-server at {self.base_url} failed: {message}")
                        for token in doc.get("tokens") or []:
                            started = True
                            token = int(token)
                            if token in self.eos:
                                return  # leaving the block closes the stream, which cancels the slot
                            yield token
                        if doc.get("stop") or cancelled():
                            return
                return
            except (httpx.RemoteProtocolError, httpx.ConnectError) as exc:
                if self.process is not None:
                    self.process.check()
                if attempt == 1 and not started and (self.process is None or self.process.running):
                    continue
                raise EngineUnavailable(self._unreachable(exc)) from exc
            except httpx.HTTPError as exc:
                if self.process is not None:
                    self.process.check()
                raise EngineUnavailable(self._unreachable(exc)) from exc

    # -- synthesis --------------------------------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
        extras: SynthesisExtras | None = None,
    ) -> Iterator[FloatArray]:
        if speed != 1.0:
            raise InvalidRequest(f"model '{self.name}' speaks at one speed only; send speed 1.0", param="speed")
        if extras is not None and extras.reference_audio is not None:
            raise InvalidRequest(f"model '{self.name}' cannot imitate a voice from reference audio", param="voice")
        speaker = self.speaker(voice)
        stop = cancelled or (lambda: False)
        if self.process is not None:
            self.process.ensure_started()
        from bolo.text.segment import chunk_text

        for piece in chunk_text(text, max_chars=self.max_chars, first_max_chars=self.max_chars) or [text]:
            if stop():
                return
            prompt = build_prompt(self.prompt_format, self.tokenize, piece, speaker)
            yield from self.stream_tokens(
                self.generate(prompt, token_budget(piece, self.max_new_tokens, self.frames_per_char), stop), stop
            )

    def stream_tokens(self, tokens: Iterator[int], stop: Callable[[], bool]) -> Iterator[FloatArray]:
        """Generated ids -> audio, decoded every ``stream_frames`` frames (bolo-torch's ``OrpheusSNAC._stream``)."""
        audio_tokens: list[int] = []
        emitted = 0  # frames already sent
        try:
            for token in tokens:
                if is_audio_token(token):
                    audio_tokens.append(token)
                    ready = len(audio_tokens) // FRAME - 1  # keep one frame of look-ahead
                    if ready - emitted >= self.stream_frames:
                        chunk = self.decode_range(frames_from_tokens(audio_tokens), emitted, ready)
                        emitted = ready
                        if chunk.size:
                            yield chunk
                if stop():
                    break
        finally:
            close = getattr(tokens, "close", None)
            if close is not None:
                close()  # ends the HTTP stream (and so the generation) when the consumer stops early
        frames = frames_from_tokens(audio_tokens)
        if len(frames) > emitted and not stop():
            chunk = self.decode_range(frames, emitted, len(frames))
            if chunk.size:
                yield chunk

    def decode_range(self, frames: list[list[int]], start: int, end: int) -> FloatArray:
        """Audio of ``frames[start:end]``, decoded with up to ``context_frames`` frames of left context."""
        end = min(end, len(frames))
        if end <= start:
            return np.zeros(0, dtype=np.float32)
        left = max(0, start - self.context_frames)
        window = frames[left:end]
        audio = self.decoder.decode(snac_levels(window))
        per_frame = len(audio) // len(window)
        return np.clip(audio[(start - left) * per_frame :], -1.0, 1.0).astype(np.float32)


def load_model(engine: TTSServerEngine, path: Path, spec: dict[str, Any], opts: LoadOptions) -> OrpheusLlamaCpp:
    """The ``orpheus-llamacpp`` kind of the tts-server engine: an operator-run or a bolo-managed llama-server."""
    from bolo.tts_runtimes import LLAMA

    s = engine._settings()
    runtime = str(spec.get("runtime") or LLAMA)
    if runtime != LLAMA:
        raise EngineUnavailable(f"the {KIND} kind runs on llama-server, not {runtime!r}")
    name = str(spec.get("model_id") or path.name)
    url = engine.remote(spec)
    decoder_path = find_decoder(path, spec, Path(s.data_dir), download=url is not None)
    threads = int(spec.get("decoder_threads") or max(1, min(2, opts.threads)))
    decoder = SnacDecoder(open_session(decoder_path, threads, opts.device))
    if url:
        model = OrpheusLlamaCpp(
            name, spec, decoder, base_url=url, api_key=s.tts_server_api_key, timeout_s=s.tts_server_timeout_s
        )
    else:
        process = engine._process(runtime, name, path, spec, opts)
        model = OrpheusLlamaCpp(name, spec, decoder, process=process, timeout_s=s.tts_server_timeout_s)
    try:
        model.connect()
    except BaseException:
        model.close()
        raise
    return model
