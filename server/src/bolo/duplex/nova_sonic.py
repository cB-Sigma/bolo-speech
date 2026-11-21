"""Amazon Nova 2.5 Sonic (Amazon Bedrock) as a full-duplex backend.

Nova Sonic listens all the time and decides itself when the caller has finished (its turn detection waits for a
pause of 1.5 to 2 s) and when it was talked over (it stops and says so), so it runs behind bolo's full-duplex bridge
(:mod:`bolo.duplex.bridge`), like Moshi: the caller's audio goes in continuously, the model's speech and text come
back as it produces them.

From AWS's documentation, read 2026-11-21: the model card (bedrock/latest/userguide/model-card-amazon-nova-2-5-sonic:
model id ``amazon.nova-2-5-sonic``, ``InvokeModelWithBidirectionalStream`` on ``bedrock-runtime`` in us-east-1,
us-west-2, eu-north-1 and ap-northeast-1, 20 concurrent sessions per account and Region), the API reference of
``InvokeModelWithBidirectionalStream`` (a session lasts at most 8 minutes; input and output are ``chunk.bytes``),
and the Nova 2 Sonic user guide (getting started, input events, output events, barge-in, turn-taking, language
support):

* in: ``sessionStart`` (``inferenceConfiguration``, ``turnDetectionConfiguration.endpointingSensitivity`` HIGH /
  MEDIUM / LOW), ``promptStart`` (``promptName``, ``textOutputConfiguration``, ``audioOutputConfiguration``: LPCM at
  24 kHz, ``voiceId``), the system prompt as a ``SYSTEM`` text content (``contentStart`` / ``textInput`` /
  ``contentEnd``), then one ``USER`` audio content (``contentStart`` with ``audioInputConfiguration``: LPCM 16 kHz,
  base64) carrying ``audioInput`` events; at the end ``contentEnd``, ``promptEnd``, ``sessionEnd``;
* out: ``completionStart``; ``contentStart`` (``role`` USER or ASSISTANT; ``additionalModelFields``
  ``generationStage`` SPECULATIVE: what the model plans to say, or FINAL: what it said), ``textOutput``,
  ``audioOutput`` (base64 LPCM), ``contentEnd`` (``stopReason`` PARTIAL_TURN, END_TURN, or INTERRUPTED when the
  caller talked over it), ``usageEvent``, ``completionEnd``.

The transport is AWS's Python SDK (``aws-sdk-bedrock-runtime``: SigV4 signing, HTTP/2 through awscrt, the event-stream
framing); bolo sends each event as ``InvokeModelWithBidirectionalStreamInputChunk(value=BidirectionalInputPayloadPart
(bytes_=<event JSON>))`` and reads ``InvokeModelWithBidirectionalStreamOutputChunk`` events back, as AWS's sample
does. The SDK needs Python 3.12 or newer: ``pip install "bolo-speech[nova]"``.

Not carried over: function tools (bolo's full-duplex bridge carries no tool calls), the user transcript Nova sends
(bolo transcribes the caller itself), and conversation history.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np

from bolo.duplex.base import (
    DuplexAudio,
    DuplexBackend,
    DuplexConfig,
    DuplexEnd,
    DuplexEvent,
    DuplexInterrupt,
    DuplexSession,
    DuplexText,
)
from bolo.engines.base import FloatArray
from bolo.errors import ComponentMissing, EngineUnavailable

log = logging.getLogger("bolo.duplex.nova")

INPUT_RATE = 16000
OUTPUT_RATE = 24000
DEFAULT_MODEL = "amazon.nova-2-5-sonic"
DEFAULT_VOICE = "matthew"
#: the voices of Nova 2 Sonic's language table (tiffany and matthew speak every supported language)
VOICES = ("tiffany", "matthew", "amy", "olivia", "kiara", "arjun", "ambre", "florian", "beatrice", "lorenzo", "tina",
          "lennart", "lupe", "carlos", "carolina", "leo")  # fmt: skip
INSTALL_HINT = 'pip install "bolo-speech[nova]" (AWS\'s aws-sdk-bedrock-runtime with awscrt; Python 3.12 or newer)'


def sdk_missing() -> str | None:
    """Why AWS's SDK cannot be used here, or None."""
    import importlib

    try:
        importlib.import_module("aws_sdk_bedrock_runtime")
        importlib.import_module("smithy_http.aio.crt")  # awscrt: HTTP/2, which the event stream needs
    except ImportError as exc:
        return f"{exc.name or 'aws-sdk-bedrock-runtime'} is not installed ({INSTALL_HINT})"
    return None


def _event(name: str, body: dict[str, Any]) -> dict[str, Any]:
    return {"event": {name: body}}


class NovaSonicSession(DuplexSession):
    input_rate = INPUT_RATE
    frame_samples = 512  # 32 ms: the pieces AWS's sample sends

    def __init__(self, stream: Any, chunk: Callable[[bytes], Any], where: str, *, tail_s: float = 1.0) -> None:
        self.stream = stream
        #: silence handed on after each stretch of the model's speech: Nova sends audio only while it speaks, and the
        #: bridge ends a response after a pause of its silence_ms in the model's audio
        self.tail_s = tail_s
        self._chunk = chunk
        self.where = where
        self.prompt = str(uuid.uuid4())
        self.audio_content = str(uuid.uuid4())
        self._events: asyncio.Queue[DuplexEvent] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None
        self._closed = False
        self._ended = False
        #: per output content id: (role, generation stage)
        self._contents: dict[str, tuple[str, str]] = {}
        self._interrupted: set[str] = set()  # completions already reported as interrupted
        self._spoke = False  # text of the current completion was handed on (later pieces get a leading space)

    async def send(self, name: str, body: dict[str, Any]) -> None:
        if self._closed:
            raise EngineUnavailable(f"{self.where}: the session is closed")
        data = json.dumps(_event(name, body)).encode("utf-8")
        try:
            await self.stream.input_stream.send(self._chunk(data))
        except Exception as exc:
            raise EngineUnavailable(f"{self.where}: the stream broke ({type(exc).__name__}: {exc})") from exc

    async def start(self, config: DuplexConfig, *, voice: str, sensitivity: str | None, max_tokens: int) -> None:
        start: dict[str, Any] = {"inferenceConfiguration": {"maxTokens": max_tokens, "topP": 0.9, "temperature": 0.7}}
        if sensitivity:
            start["turnDetectionConfiguration"] = {"endpointingSensitivity": sensitivity}
        await self.send("sessionStart", start)
        await self.send(
            "promptStart",
            {
                "promptName": self.prompt,
                "textOutputConfiguration": {"mediaType": "text/plain"},
                "audioOutputConfiguration": {
                    "mediaType": "audio/lpcm", "sampleRateHertz": OUTPUT_RATE, "sampleSizeBits": 16,
                    "channelCount": 1, "voiceId": voice, "encoding": "base64", "audioType": "SPEECH",
                },
            },
        )  # fmt: skip
        if config.instructions:
            name = str(uuid.uuid4())
            await self.send(
                "contentStart",
                {"promptName": self.prompt, "contentName": name, "type": "TEXT", "interactive": False,
                 "role": "SYSTEM", "textInputConfiguration": {"mediaType": "text/plain"}},
            )  # fmt: skip
            await self.send(
                "textInput", {"promptName": self.prompt, "contentName": name, "content": config.instructions}
            )
            await self.send("contentEnd", {"promptName": self.prompt, "contentName": name})
        await self.send(
            "contentStart",
            {
                "promptName": self.prompt, "contentName": self.audio_content, "type": "AUDIO", "interactive": True,
                "role": "USER",
                "audioInputConfiguration": {
                    "mediaType": "audio/lpcm", "sampleRateHertz": INPUT_RATE, "sampleSizeBits": 16,
                    "channelCount": 1, "audioType": "SPEECH", "encoding": "base64",
                },
            },
        )  # fmt: skip
        self._reader = asyncio.create_task(self._read(), name="bolo-nova-sonic")

    async def push(self, samples: FloatArray) -> None:
        if not len(samples) or self._closed:
            return
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        for i in range(0, len(pcm), self.frame_samples * 2):
            content = base64.b64encode(pcm[i : i + self.frame_samples * 2]).decode("ascii")
            await self.send(
                "audioInput", {"promptName": self.prompt, "contentName": self.audio_content, "content": content}
            )

    async def _read(self) -> None:
        end = DuplexEnd("ended")
        try:
            _, output = await self.stream.await_output()
            async for item in output:
                value = getattr(item, "value", None)
                data = getattr(value, "bytes_", None)
                if data is None:  # one of the stream's exception members (validation, throttling, timeout, ...)
                    name = type(item).__name__.replace("InvokeModelWithBidirectionalStreamOutput", "")
                    message = getattr(value, "message", None) or str(value or name)
                    end = DuplexEnd("failed", f"{name}: {message}")
                    break
                self._handle(json.loads(bytes(data).decode("utf-8")).get("event") or {})
        except asyncio.CancelledError:
            end = DuplexEnd("closed")
            raise
        except Exception as exc:
            end = DuplexEnd("closed") if self._closed else DuplexEnd("failed", f"{type(exc).__name__}: {exc}")
        finally:
            if self._closed:
                end = DuplexEnd("closed")
            self._finish(end)

    def _finish(self, end: DuplexEnd) -> None:
        if not self._ended:
            self._ended = True
            self._events.put_nowait(end)

    def _handle(self, event: dict[str, Any]) -> None:
        if "completionStart" in event:
            self._spoke = False
        elif "contentStart" in event:
            start = event["contentStart"]
            stage = ""
            with contextlib.suppress(ValueError, TypeError):
                stage = str(json.loads(start.get("additionalModelFields") or "{}").get("generationStage") or "")
            self._contents[str(start.get("contentId"))] = (str(start.get("role") or ""), stage)
        elif "textOutput" in event:
            out = event["textOutput"]
            role, stage = self._contents.get(str(out.get("contentId")), ("", ""))
            # what the model is about to say, as it plans it (the FINAL text repeats what was spoken afterwards)
            if role == "ASSISTANT" and stage != "FINAL" and out.get("content"):
                text = str(out["content"])
                self._events.put_nowait(DuplexText((" " if self._spoke else "") + text))
                self._spoke = True
        elif "audioOutput" in event:
            content = event["audioOutput"].get("content")
            if content:
                pcm = base64.b64decode(content)
                x = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768.0
                self._events.put_nowait(DuplexAudio(x, OUTPUT_RATE))
        elif "contentEnd" in event:
            end = event["contentEnd"]
            completion = str(end.get("completionId") or end.get("contentId"))
            if end.get("stopReason") == "INTERRUPTED" and completion not in self._interrupted:
                self._interrupted.add(completion)
                self._events.put_nowait(DuplexInterrupt())
            if end.get("type") == "AUDIO":  # the model stopped speaking: the pause that ends the response
                self._events.put_nowait(DuplexAudio(np.zeros(int(self.tail_s * OUTPUT_RATE), np.float32), OUTPUT_RATE))

    async def events(self) -> AsyncIterator[DuplexEvent]:
        while True:
            event = await self._events.get()
            yield event
            if isinstance(event, DuplexEnd):
                return

    async def close(self) -> None:
        if self._closed:
            return
        with contextlib.suppress(Exception):
            await self.send("contentEnd", {"promptName": self.prompt, "contentName": self.audio_content})
            await self.send("promptEnd", {"promptName": self.prompt})
            await self.send("sessionEnd", {})
        self._closed = True
        with contextlib.suppress(Exception):
            await self.stream.input_stream.close()
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
        self._finish(DuplexEnd("closed"))


class NovaSonicBackend(DuplexBackend):
    """Nova Sonic sessions through Amazon Bedrock (AWS's SDK). ``stream_factory`` stands in for the SDK in tests:
    an async callable that opens a bidirectional stream and returns it with the function that wraps an event's
    bytes as an input chunk."""

    name = "nova-sonic"

    def __init__(
        self,
        *,
        model: str,
        model_id: str = DEFAULT_MODEL,
        region: str = "us-east-1",
        endpoint_uri: str | None = None,
        credentials: dict[str, str] | None = None,
        provider: str = "bedrock",
        max_sessions: int = 20,
        default_voice: str = DEFAULT_VOICE,
        sensitivity: str | None = None,
        max_tokens: int = 1024,
        tail_s: float = 1.0,
        stream_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.model = model
        self.model_id = model_id
        self.region = region
        self.endpoint_uri = endpoint_uri or f"https://bedrock-runtime.{region}.amazonaws.com"
        self.credentials = dict(credentials or {})
        self.provider = provider
        self.max_sessions = max_sessions
        self.default_voice = default_voice
        self.sensitivity = sensitivity
        self.max_tokens = max_tokens
        self.tail_s = tail_s
        self.notices = []
        self._stream_factory = stream_factory
        self._open_sessions = 0

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(cloud=True, provider=self.provider, region=self.region, model_id=self.model_id)
        return out

    async def _open_stream(self) -> tuple[Any, Callable[[bytes], Any]]:
        if self._stream_factory is not None:
            return await self._stream_factory()  # type: ignore[no-any-return]
        problem = sdk_missing()
        if problem:
            raise ComponentMissing(f"Amazon Nova Sonic needs AWS's SDK: {problem}")
        from aws_sdk_bedrock_runtime.aio.client import AsyncBedrockRuntimeClient  # type: ignore[import-not-found]
        from aws_sdk_bedrock_runtime.aio.config import AsyncBedrockRuntimeConfig  # type: ignore[import-not-found]
        from aws_sdk_bedrock_runtime.models import (  # type: ignore[import-not-found]
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk,
            InvokeModelWithBidirectionalStreamOperationInput,
        )
        from smithy_http.aio.crt import AWSCRTHTTPClient  # type: ignore[import-not-found]

        overrides: dict[str, Any] = {"endpoint_uri": self.endpoint_uri, "region": self.region}
        overrides["transport"] = AWSCRTHTTPClient()  # event streams need HTTP/2
        for key in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
            if self.credentials.get(key):
                overrides[key] = self.credentials[key]
        config = await AsyncBedrockRuntimeConfig.resolve(**overrides)
        client = AsyncBedrockRuntimeClient(config=config)
        stream = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )

        def chunk(data: bytes) -> Any:
            return InvokeModelWithBidirectionalStreamInputChunk(value=BidirectionalInputPayloadPart(bytes_=data))

        return stream, chunk

    async def open(self, config: DuplexConfig) -> DuplexSession:
        if self._open_sessions >= self.max_sessions:
            raise EngineUnavailable(
                f"{self.model} is serving {self._open_sessions} sessions, the most it is set to hold "
                "(Amazon Bedrock allows 20 per account and Region by default)"
            )
        voice = (config.voice or "").lower()
        voice = voice if voice in VOICES else self.default_voice
        try:
            stream, chunk = await self._open_stream()
        except (ComponentMissing, EngineUnavailable):
            raise
        except Exception as exc:  # credentials, network, the service refusing the call
            raise EngineUnavailable(
                f"Amazon Bedrock ({self.region}) did not open a Nova Sonic stream: {type(exc).__name__}: {exc}"
            ) from exc
        session = NovaSonicSession(
            stream, chunk, f"Amazon Bedrock ({self.region}, {self.model_id})", tail_s=self.tail_s
        )
        sensitivity = str(config.options.get("endpointing_sensitivity") or self.sensitivity or "") or None
        await session.start(config, voice=voice, sensitivity=sensitivity, max_tokens=self.max_tokens)
        self._open_sessions += 1
        original_close = session.close

        async def close() -> None:
            if not session._closed:
                self._open_sessions = max(0, self._open_sessions - 1)
            await original_close()

        session.close = close  # type: ignore[method-assign]
        return session


def bedrock_settings(provider: Any) -> tuple[str, str, dict[str, str]]:
    """(region, endpoint, credentials) of a provider of kind ``bedrock``. The key is ``<access key id>:<secret access
    key>`` (optionally ``:<session token>``); without one, AWS's SDK finds credentials itself (``AWS_ACCESS_KEY_ID``
    and ``AWS_SECRET_ACCESS_KEY`` in the environment, or a profile)."""
    import re

    base = str(getattr(provider, "base_url", "") or "")
    match = re.match(r"https://bedrock-runtime\.([a-z0-9-]+)\.amazonaws\.com", base)
    options = dict(getattr(provider, "options", {}) or {})
    region = str(options.get("region") or (match.group(1) if match else "us-east-1"))
    # an AWS address follows the Region; any other base_url (a proxy, a VPC endpoint) is used as it is
    endpoint = f"https://bedrock-runtime.{region}.amazonaws.com" if match or not base else base
    credentials: dict[str, str] = {}
    key = str(getattr(provider, "api_key", "") or "")
    if key:
        parts = key.split(":", 2)
        if len(parts) >= 2:
            credentials["aws_access_key_id"], credentials["aws_secret_access_key"] = parts[0], parts[1]
            if len(parts) == 3 and parts[2]:
                credentials["aws_session_token"] = parts[2]
        elif options.get("access_key_id"):
            credentials["aws_access_key_id"], credentials["aws_secret_access_key"] = str(options["access_key_id"]), key
    return region, endpoint, credentials
