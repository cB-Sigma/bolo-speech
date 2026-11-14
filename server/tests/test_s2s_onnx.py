"""LFM3-Audio in ONNX Runtime (bolo.s2s_onnx) with scripted stand-ins for the graphs and a toy tokenizer: no model
files, no onnxruntime, no network. The real model is exercised by tests/test_real.py (opt-in)."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import websockets

import bolo.s2s_onnx as m
from bolo.config import Settings
from bolo.core import Bolo
from bolo.engines.base import LoadOptions, SynthesisExtras
from bolo.errors import ComponentMissing, EngineUnavailable
from bolo.s2s import AudioDelta, ResponseEnd, S2SConfig, TextDelta, ToolCall
from conftest import LiveServer
from test_realtime import b64_pcm24

HIDDEN = 8
CONTROL = ["<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|audio|>", "<|audio_out|>", "<|audio_start|>",
           "<|text_end|>", "<|mixed_start|>"]  # fmt: skip
KEPT = ["<tool_call>", "</tool_call>", "<think>", "</think>"]  # added tokens that are not "special": decoded
SPECIAL = {name: 1000 + i for i, name in enumerate(CONTROL + KEPT)}
NAMES = {v: k for k, v in SPECIAL.items()}
VOCAB = 1100
_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL) + ")")


def encode(text: str) -> list[int]:
    ids: list[int] = []
    for part in _SPLIT.split(text):
        if part in SPECIAL:
            ids.append(SPECIAL[part])
        else:
            ids += [ord(c) for c in part]
    return ids


def decode(ids: list[int], *, raw: bool = False) -> str:
    return "".join(
        NAMES[i] if (raw or NAMES.get(i) in KEPT) and i in NAMES else chr(i)
        for i in ids
        if raw or i < 1000 or NAMES.get(i) in KEPT
    )


TOKENS = m.Tokens.from_lookup(SPECIAL.get)


class FakeGraphs:
    """Deterministic graphs. The decoder "generates" the scripted text tokens, one per text slot; the depth
    decoder gives frame ``k`` the codes ``(k * 7 + codebook) % 2016`` and ends the speech after the scripted
    number of frames; the codec's rows depend on their own frame only."""

    def __init__(self, script: list[int] | None = None, frames_before_eos: int = 10**9) -> None:
        self.script = list(script or [])
        self.frames_before_eos = frames_before_eos
        self.frames = 0
        self.fed: list[tuple[str, int]] = []  # every position given to the decoder: (kind, id or frame index)
        self.audio_rows: list[np.ndarray] = []
        self.codes_fed: list[np.ndarray] = []
        self.encode_calls = 0
        self.depth_calls = 0
        self.codec_calls: list[int] = []

    def empty_state(self) -> m.DecoderState:
        return m.DecoderState({}, 0)

    def encode_audio(self, features: np.ndarray) -> np.ndarray:
        self.encode_calls += 1
        n = -(features.shape[1] // -8)
        padded = np.pad(features, ((0, 0), (0, n * 8 - features.shape[1])))
        blocks = padded.reshape(features.shape[0], n, 8).mean(axis=(0, 2))  # causal: a row sees its own 8 frames
        return np.repeat(blocks[:, None], HIDDEN, axis=1).astype(np.float32)

    def decode(self, state, input_ids, audio_features=None, audio_out_codes=None):
        rows = iter(audio_features if audio_features is not None else [])
        frames = iter(audio_out_codes if audio_out_codes is not None else [])
        for token in input_ids:
            if token == TOKENS.audio:
                self.audio_rows.append(next(rows))
                self.fed.append(("audio", len(self.audio_rows) - 1))
            elif token == TOKENS.audio_out:
                self.codes_fed.append(next(frames))
                self.fed.append(("frame", len(self.codes_fed) - 1))
            else:
                self.fed.append(("text", int(token)))
        assert next(rows, None) is None and next(frames, None) is None
        state.length += len(input_ids)
        logits = np.zeros(VOCAB, dtype=np.float32)
        logits[self.script[0] if self.script else TOKENS.im_end] = 10.0
        return logits, np.full(HIDDEN, float(state.length), dtype=np.float32)

    def depth(self, hidden, codes, masked):
        self.depth_calls += 1
        logits = np.zeros((12, 2017), dtype=np.float32)
        for index in range(12):
            target = 2016 if self.frames >= self.frames_before_eos else (self.frames * 7 + index) % 2016
            logits[index, target] = 1e4
        if not masked.any() or masked.sum() == 4:  # the last pass of the 4-4-4 schedule
            self.frames += 1
        return logits

    def codec(self, codes):
        self.codec_calls.append(len(codes))
        rows = np.zeros((len(codes) * 6, 1282), dtype=np.float32)
        for i, frame in enumerate(codes):
            for j in range(6):
                rows[i * 6 + j, :641] = -3.0 + 0.001 * (frame[0] % 50)  # log-magnitudes
                rows[i * 6 + j, 641:] = 0.1 * j + 0.01 * (frame[1] % 30) * np.arange(641) / 641  # phases
        return rows


class TextGraphs(FakeGraphs):
    """The decoder consumes its script as the text slots are sampled (the session samples from the last logits)."""

    def decode(self, state, input_ids, audio_features=None, audio_out_codes=None):
        if len(input_ids) == 1 and self.script and input_ids[0] == self.script[0]:
            self.script.pop(0)
        return super().decode(state, input_ids, audio_features, audio_out_codes)


def make(script: str | list[int] = "", frames_before_eos: int = 10**9, config=None, **options):
    ids = encode(script) if isinstance(script, str) else script
    graphs = TextGraphs(ids, frames_before_eos)
    model = m.Lfm3Model(graphs, encode, decode, TOKENS)
    return graphs, m.Lfm3Session(model, config, seed=0, **options)


def prompt_of(graphs: FakeGraphs) -> str:
    out = ""
    for kind, value in graphs.fed:
        out += "<|audio|>" if kind == "audio" else "<|audio_out|>" if kind == "frame" else decode([value], raw=True)
    return out


def collect(events):
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    audio = [e.samples for e in events if isinstance(e, AudioDelta)]
    calls = [e for e in events if isinstance(e, ToolCall)]
    end = events[-1]
    assert isinstance(end, ResponseEnd) and sum(isinstance(e, ResponseEnd) for e in events) == 1
    return text, (np.concatenate(audio) if audio else np.zeros(0, np.float32)), calls, end


# -- front end ---------------------------------------------------------------------------------------------------
def test_log_mel_frames_and_padding():
    fe = m.FeatureSettings()
    mel = m.LogMel(fe)
    assert mel(np.zeros(100, np.float32)).shape == (128, 1)  # shorter than a window: padded to one frame
    assert mel(np.zeros(400, np.float32)).shape == (128, 1)
    assert mel(np.zeros(401, np.float32)).shape == (128, 2)  # no trailing sample is dropped
    assert mel(np.zeros(16000, np.float32)).shape == (128, 99)
    assert mel(np.zeros(16000, np.float32), pad=False).shape == (128, 98)
    assert mel(np.zeros(399, np.float32), pad=False).shape == (128, 0)
    np.testing.assert_allclose(mel(np.zeros(800, np.float32)), -2.0)  # (floor + 4) / 4
    assert fe.num_tokens(99) == 13 and fe.num_tokens(96) == 12


def test_log_mel_values():
    t = np.arange(16000) / 16000.0
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    feats = m.LogMel(m.FeatureSettings())(tone)
    filters = m.mel_filter_bank(201, 128, 16000)
    assert filters.shape == (201, 128) and (filters >= 0).all()
    band = int(feats[:, 10].argmax())
    centre_hz = np.arange(201)[filters[:, band].argmax()] * 40.0
    assert abs(centre_hz - 1000.0) <= 40.0
    # power of a 0.5 amplitude tone under a periodic Hann window: (0.5 * 400 / 4)^2 at the peak bin
    peak = np.log10(((0.5 * 400 / 4) ** 2) * filters[25, band])
    assert abs(feats[band, 10] - (peak + 4.0) / 4.0) < 0.02
    np.testing.assert_allclose(feats[:, 5], feats[:, 50], atol=2e-3)  # stationary


def test_streamed_log_mel_is_a_prefix():
    rng = np.random.default_rng(1)
    clip = rng.standard_normal(16000 * 2).astype(np.float32) * 0.1
    mel = m.LogMel(m.FeatureSettings())
    whole = mel(clip)
    part = mel(clip[:9000], pad=False)
    np.testing.assert_allclose(part, whole[:, : part.shape[1]], atol=1e-6)


# -- sampling ----------------------------------------------------------------------------------------------------
def test_sample_greedy_penalty_and_masks():
    rng = np.random.default_rng(0)
    logits = np.array([1.0, 3.0, 2.9, -1.0], dtype=np.float32)
    assert m.sample(logits, rng) == 1
    assert m.sample(logits, rng, repetition_penalty=1.1, history=[1, 1]) == 2  # 3.0 / 1.1 < 2.9
    assert m.sample(logits, rng, suppress=[1, 2]) == 0
    assert m.sample(logits, rng, allowed=[0, 3]) == 0
    penalised = np.array([-1.0, -1.05], dtype=np.float32)
    assert m.sample(penalised, rng, repetition_penalty=1.1, history=[0]) == 1  # negative scores are multiplied


def test_sample_top_k_only_draws_the_best():
    rng = np.random.default_rng(0)
    logits = np.linspace(0.0, 5.0, 50).astype(np.float32)
    draws = {m.sample(logits, rng, temperature=1.0, top_k=3) for _ in range(200)}
    assert draws == {47, 48, 49}


def test_audio_frame_uses_three_passes_and_reveals_all():
    graphs = FakeGraphs()
    codes = m.sample_audio_frame(graphs, np.zeros(HIDDEN, np.float32), m.GenerationSettings(), np.random.default_rng(0))
    assert graphs.depth_calls == 3
    assert codes.tolist() == [i % 2016 for i in range(12)]


# -- codec -------------------------------------------------------------------------------------------------------
def test_codec_stream_equals_one_shot():
    frames = [np.array([(k * 7 + i) % 2016 for i in range(12)]) for k in range(23)]
    settings = m.CodecSettings()
    whole = m.CodecStream(FakeGraphs(), settings)
    for f in frames:
        whole.push(f)
    reference = whole.decode(final=True)
    assert len(reference) == 23 * 1920

    graphs = FakeGraphs()
    stream = m.CodecStream(graphs, settings, context=4)
    parts = []
    for k, f in enumerate(frames):
        stream.push(f)
        if k in (1, 7, 8, 19):
            parts.append(stream.decode())
    parts.append(stream.decode(final=True))
    np.testing.assert_allclose(np.concatenate(parts), reference, atol=1e-5)
    assert len(parts[0]) == 2 * 1920 - 480  # the first chunk lacks the trim and the held-back overlap
    assert graphs.codec_calls == [2, 2 + 6, 4 + 1, 4 + 11, 4 + 3]  # each chunk behind its context
    assert len(stream.decode(final=True)) == 0  # nothing left


# -- chat format -------------------------------------------------------------------------------------------------
TOOL = {"type": "function", "name": "set_volume", "description": "Set the volume.",
        "parameters": {"type": "object", "properties": {"volume": {"type": "integer"}}, "required": ["volume"]}}  # fmt: skip


def test_system_prompt():
    assert m.system_prompt(None, []) == m.SYSTEM_MIXED
    assert m.system_prompt("Be brief.", []) == m.SYSTEM_MIXED + "\n\nBe brief."
    assert m.system_prompt(None, [], speech=False) == ""
    with_tools = m.system_prompt(None, [TOOL])
    assert with_tools.startswith(
        "Respond with function calls, and interleaved text and audio.\n\nList of tools:\n\n<tools>\n"
    )
    line = with_tools.split("<tools>\n")[1].split("\n</tools>")[0]
    assert json.loads(line) == {k: v for k, v in TOOL.items() if k != "type"}
    nested = {"type": "function", "function": {k: v for k, v in TOOL.items() if k != "type"}}
    assert m.system_prompt(None, [nested]) == with_tools


def test_tool_call_parsing():
    text = "<tool_call>\n<function=set_volume>\n<parameter=volume>\n30\n</parameter>\n<parameter=room>\nkitchen\n</parameter>\n</function>\n</tool_call>"
    assert m.parse_tool_calls(text) == [("set_volume", {"volume": 30, "room": "kitchen"})]
    parser = m.OutputParser()
    said, calls = "", []
    for piece in ["<think>", "hm", "</think>", "\n\n", "<tool_call>", "\n<function=a>\n</function>\n", "</tool_call>", "\n\n",
                  "<tool_call>", "<function=b>\n<parameter=x>\n[1, 2]\n</parameter></function>", "</tool_call>", " Ok", "."]:  # fmt: skip
        s, c = parser.feed(piece)
        said += s
        calls += c
    assert calls == [("a", {}), ("b", {"x": [1, 2]})]
    assert said == "Ok."


# -- the response ------------------------------------------------------------------------------------------------
def test_interleaved_response_follows_the_mode_machine():
    # 6 text tokens, 12 frames, 5 text tokens and <|text_end|>, then speech until the end-of-audio frame
    graphs, session = make("Hello world<|text_end|><|im_end|>", frames_before_eos=20)
    session.append_audio(np.zeros(16000, np.float32))
    session.commit()
    events = list(session.respond(modalities=["text", "audio"]))
    text, audio, calls, end = collect(events)
    assert text == "Hello world" and not calls and end.status == "completed"
    prompt = prompt_of(graphs)
    head = "<|startoftext|><|im_start|>system\n" + m.SYSTEM_MIXED + "<|im_end|>\n<|im_start|>user\n" + "<|audio|>" * 13
    head += "<|im_end|>\n<|im_start|>assistant\n<|mixed_start|>"
    assert prompt.startswith(head)
    body = prompt[len(head) :]
    assert (
        body == "Hello " + "<|audio_out|>" * 12 + "world<|text_end|>" + "<|audio_out|>" * 9
    )  # 8 frames + the end frame
    assert len(graphs.codes_fed) == 21 and (graphs.codes_fed[-1] == 2016).all()
    assert graphs.codes_fed[3].tolist() == [(3 * 7 + i) % 2016 for i in range(12)]
    assert len(audio) == 20 * 1920
    assert end.usage == {
        "input_tokens": len(encode(head)),
        "output_tokens": 12 + 20,
        "total_tokens": len(encode(head)) + 32,
    }
    # the audio came in chunks: 2 frames first, then 6 at a time, flushed at the end of a block and of the speech
    assert graphs.codec_calls[0] == 2 and len(graphs.codec_calls) >= 4
    assert session.stats.audio_frames == 20 and session.stats.text_tokens == 12


def test_next_turn_continues_the_context():
    graphs, session = make("Hi.<|text_end|><|im_end|>", frames_before_eos=3)
    session.add_text("  Hello  ")
    list(session.respond(modalities=["text", "audio"]))
    first = len(graphs.fed)
    graphs.script = encode("Yes.<|im_end|>")
    session.add_text("Again?")
    text, audio, _, end = collect(list(session.respond(modalities=["text"])))
    assert text == "Yes." and len(audio) == 0 and end.status == "completed"
    tail = "".join("<|audio_out|>" if k == "frame" else decode([v], raw=True) for k, v in graphs.fed[first:])
    assert tail == "<|im_end|>\n<|im_start|>user\nAgain?<|im_end|>\n<|im_start|>assistant\n<think></think>Yes."
    assert prompt_of(graphs).count("<|startoftext|>") == 1  # nothing was evaluated twice


def test_text_only_never_switches_to_speech():
    graphs, session = make([TOKENS.mixed_start] + encode("x<|im_end|>"))
    session.add_text("Hello")
    # the scripted logits prefer <|mixed_start|>; a text response suppresses it, so the arg-max is something else
    events = list(session.respond(modalities=["text"], max_output_tokens=3))
    assert not any(isinstance(e, AudioDelta) for e in events)
    assert ("text", TOKENS.mixed_start) not in graphs.fed[-3:]
    assert events[-1].status == "incomplete"


def test_tool_call_then_result_then_answer():
    config = S2SConfig(tools=[TOOL])
    call = (
        "<tool_call>\n<function=set_volume>\n<parameter=volume>\n30\n</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    graphs, session = make(call, config=config)
    session.append_audio(np.zeros(8000, np.float32))
    events = list(session.respond(modalities=["text", "audio"]))  # commits the audio itself
    text, audio, calls, end = collect(events)
    assert text == "" and len(audio) == 0 and end.status == "completed"
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [("set_volume", {"volume": 30})]
    assert calls[0].call_id.startswith("call_")
    assert "<tools>\n" in prompt_of(graphs) and "<|mixed_start|>" not in prompt_of(graphs)

    first = len(graphs.fed)
    graphs.script = [TOKENS.mixed_start] + encode("Done.<|text_end|><|im_end|>")
    graphs.frames_before_eos = graphs.frames + 2
    session.add_tool_output(calls[0].call_id, '{"volume": 30}')
    text, audio, calls, end = collect(list(session.respond(modalities=["text", "audio"])))
    assert text == "Done." and len(audio) == 2 * 1920 and not calls
    tail = "".join("<|audio_out|>" if k == "frame" else decode([v], raw=True) for k, v in graphs.fed[first:])
    assert tail.startswith(
        '<|im_end|>\n<|im_start|>tool\n<tool_response>\n{"volume": 30}\n</tool_response><|im_end|>\n'
        "<|im_start|>assistant\n<|mixed_start|>Done.<|text_end|>"
    )


def test_first_token_is_a_call_or_speech_with_tools():
    config = S2SConfig(tools=[TOOL])
    graphs, session = make("Plain text<|im_end|>", config=config)  # the logits prefer plain text
    session.add_text("Louder")
    list(session.respond(modalities=["text", "audio"], max_output_tokens=1))
    assert graphs.fed[-1] in (("text", TOKENS.tool_call), ("text", TOKENS.mixed_start))
    graphs, session = make("Plain text<|im_end|>", config=config)
    session.add_text("Louder")
    list(session.respond(modalities=["text", "audio"], max_output_tokens=1, tool_choice="required"))
    assert graphs.fed[-1] == ("text", TOKENS.tool_call)
    graphs, session = make("Hi<|im_end|>", config=config)
    session.add_text("Louder")
    list(session.respond(modalities=["text", "audio"], max_output_tokens=1, tool_choice="none"))
    assert ("text", TOKENS.mixed_start) in graphs.fed  # prefilled: a spoken answer


def test_cancel_between_steps_and_recover():
    graphs, session = make("A long answer that goes on.<|text_end|><|im_end|>", frames_before_eos=100)
    session.add_text("Hello")
    seen = 0

    def cancelled() -> bool:
        return seen >= 10

    events = []
    for event in session.respond(modalities=["text", "audio"], cancelled=cancelled):
        events.append(event)
        seen += 1
    text, audio, _, end = collect(events)
    assert end.status == "cancelled"
    assert len(audio) == session.stats.audio_frames * 1920  # what was generated is flushed, tail included
    fed = len(graphs.fed)
    graphs.script = encode("Ok.<|im_end|>")
    session.add_text("Stop")
    text, _, _, end = collect(list(session.respond(modalities=["text"])))
    assert end.status == "completed" and text == "Ok."
    tail = "".join("<|audio_out|>" if k == "frame" else decode([v], raw=True) for k, v in graphs.fed[fed:])
    assert tail.startswith("<|im_end|>\n<|im_start|>user\nStop<|im_end|>\n<|im_start|>assistant\n")


def test_respond_without_input_fails():
    _, session = make("x")
    events = list(session.respond(modalities=["text"]))
    assert len(events) == 1 and events[0].status == "failed"


def test_prefill_while_listening_equals_one_shot():
    rng = np.random.default_rng(3)
    clip = (rng.standard_normal(16000 * 3 + 777) * 0.1).astype(np.float32)
    graphs_a, one_shot = make("x<|im_end|>", prefill_s=0.0)
    one_shot.append_audio(clip)
    one_shot.commit()
    list(one_shot.respond(modalities=["text"]))
    graphs_b, streamed = make("x<|im_end|>", prefill_s=0.48)
    for begin in range(0, len(clip), 1600):
        streamed.append_audio(clip[begin : begin + 1600])
    streamed.commit()
    list(streamed.respond(modalities=["text"]))
    assert graphs_a.encode_calls == 1 and graphs_b.encode_calls > 3
    assert graphs_a.fed == graphs_b.fed
    np.testing.assert_allclose(np.stack(graphs_a.audio_rows), np.stack(graphs_b.audio_rows), atol=1e-6)
    f = m.FeatureSettings()
    assert len(graphs_a.audio_rows) == f.num_tokens(f.complete_frames(f.pad_length(len(clip))))


def test_context_limit_starts_over():
    graphs, session = make("Hi<|im_end|>", max_context=90)
    session.add_text("One")
    list(session.respond(modalities=["text"]))
    graphs.script = encode("Hi<|im_end|>")
    session.add_text("Two")
    list(session.respond(modalities=["text"]))
    assert session.context_tokens > 90
    graphs.script = encode("Hi<|im_end|>")
    session.add_text("Three")
    assert session.state.length == 0  # a fresh context
    list(session.respond(modalities=["text"]))
    assert prompt_of(graphs).count("<|startoftext|>") == 2


# -- the asynchronous interface ----------------------------------------------------------------------------------
def test_conversation_streams_and_cancels():
    async def scenario() -> None:
        graphs = TextGraphs(encode("One two three four five six seven.<|text_end|><|im_end|>"), 10**9)
        backend = m.OnnxS2SBackend("unused", model=m.Lfm3Model(graphs, encode, decode, TOKENS), options={"seed": 0})
        conversation = await backend.open(S2SConfig(instructions="Be brief."))
        assert conversation.output_rate == 24000
        for _ in range(10):
            await conversation.append_audio(np.zeros(1600, np.float32))
        await conversation.commit()
        stream = conversation.respond(modalities=["text", "audio"])
        got = []
        async for event in stream:
            got.append(event)
            if sum(isinstance(e, AudioDelta) for e in got) >= 2:
                break
        await stream.aclose()  # barge-in
        assert any(isinstance(e, TextDelta) for e in got)
        # the next response comes after the cancelled one has stopped, on the same context
        graphs.script = encode("Yes.<|im_end|>")
        await conversation.add_text("And?")
        events = [e async for e in conversation.respond(modalities=["text"])]
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Yes."
        assert events[-1].status == "completed"
        assert conversation.session.stats.status == "completed"
        # an error while taking input is reported by the next response
        await conversation.add_tool_output("call_x", "{}")
        await conversation.add_tool_output("call_y", "{}")
        await conversation.add_text("x")
        await conversation.close()
        await backend.close()

    asyncio.run(scenario())


def test_conversation_reports_input_errors():
    async def scenario() -> None:
        graphs = TextGraphs(encode("Hi<|im_end|>"))
        backend = m.OnnxS2SBackend("unused", model=m.Lfm3Model(graphs, encode, decode, TOKENS))
        conversation = await backend.open(S2SConfig())
        await conversation.add_tool_output("call_x", "{}")  # no response made a call
        events = [e async for e in conversation.respond(modalities=["text"])]
        assert len(events) == 1 and events[0].status == "failed" and "tool output" in (events[0].error or "")
        await conversation.add_text("Hello")
        events = [e async for e in conversation.respond(modalities=["text"])]
        assert events[-1].status == "completed"
        await conversation.close()

    asyncio.run(scenario())


# -- text to speech ----------------------------------------------------------------------------------------------
def test_tts_prompt_and_voice():
    assert m.tts_prompt(" Hello. ", "Male", "en-GB") == (
        "<|startoftext|><|im_start|>system\nPerform TTS in British English using a male voice.<|im_end|>\n"
        "<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>"
    )
    assert "in American English using a female voice" in m.tts_prompt("x")
    assert "in Hindi using a female voice" in m.tts_prompt("x", "robot", "Hindi")  # an unknown language as given
    assert "in German " in m.tts_prompt("x", None, "de-AT")


def test_tts_model_speaks_until_the_end_frame():
    graphs = FakeGraphs(frames_before_eos=9)
    model = m.Lfm3Tts(m.Lfm3Model(graphs, encode, decode, TOKENS), {"seed": 1})
    assert model.sample_rate == 24000 and [v.id for v in model.voices] == ["female", "male"]
    chunks = list(model.synthesize("Hello there.", "male", extras=SynthesisExtras(language="fr")))
    assert sum(map(len, chunks)) == 9 * 1920 and len(chunks) == 3  # 2 frames, 6 frames, the rest
    assert "Perform TTS in French using a male voice." in prompt_of(graphs)
    assert prompt_of(graphs).endswith("<|audio_start|>" + "<|audio_out|>" * 9)
    assert list(model.synthesize("  ")) == []
    # cancellation between frames; a budget when the model never ends
    graphs = FakeGraphs()
    model = m.Lfm3Tts(m.Lfm3Model(graphs, encode, decode, TOKENS))
    seen: list[Any] = []
    for chunk in model.synthesize("Hello", cancelled=lambda: len(seen) >= 1):
        seen.append(chunk)
    assert len(seen) == 1
    audio = np.concatenate(list(m.synthesize(m.Lfm3Model(FakeGraphs(), encode, decode, TOKENS), "Hi", max_frames=7)))
    assert len(audio) == 7 * 1920


# -- files, packages and the server ------------------------------------------------------------------------------
def test_missing_files_and_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    have = {name: importlib.util.find_spec(name) is not None for name in ("onnxruntime", "tokenizers")}
    assert m.missing_packages() == [name for name, found in have.items() if not found]
    # without the package: what to install (424); with it: which file is missing (503)
    with pytest.raises(EngineUnavailable if have["tokenizers"] else ComponentMissing) as error:
        m.Lfm3Model.from_dir(tmp_path)
    assert ("tokenizer.json is missing" if have["tokenizers"] else "bolo-speech[s2s]") in str(error.value)
    with pytest.raises(EngineUnavailable if have["onnxruntime"] else ComponentMissing) as error:
        m.OnnxGraphs(tmp_path)
    assert ("decoder_q4.onnx is missing" if have["onnxruntime"] else "bolo-speech[s2s]") in str(error.value)
    assert error.value.status == (503 if have["onnxruntime"] else 424)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "tokenizers" else real(name, *a))
    assert "tokenizers" in m.missing_packages()


def fake_model(script: str = "Hello there.<|text_end|><|im_end|>", frames: int = 14) -> m.Lfm3Model:
    return m.Lfm3Model(TextGraphs(encode(script), frames), encode, decode, TOKENS)


@pytest.fixture
def onnx_server(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[LiveServer, list[m.Lfm3Model]]]:
    """A server whose speech-to-speech model is lfm3-audio-2b-onnx, "installed", with scripted graphs."""
    settings.s2s_model = "lfm3-audio-2b-onnx"
    models: list[m.Lfm3Model] = []

    def shared(path: Path, options: dict[str, Any], threads: int, device: str = "cpu") -> m.Lfm3Model:
        assert path.name == "lfm3-audio-2b-onnx" and options["precision"] == "q4" and threads >= 1
        models.append(fake_model())
        return models[-1]

    monkeypatch.setattr(m, "shared_model", shared)
    monkeypatch.setattr(m, "missing_packages", lambda: [])
    srv = LiveServer(settings)
    monkeypatch.setattr(srv.core.store, "is_installed", lambda model_id: model_id == "lfm3-audio-2b-onnx")
    srv.start()
    try:
        yield srv, models
    finally:
        srv.stop()


async def _until(ws: Any, last: str, timeout: float = 30.0) -> list[dict[str, Any]]:
    events = []
    while True:
        event = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        events.append(event)
        if event["type"] == last:
            return events


async def test_realtime_session_with_the_in_process_model(onnx_server: tuple[LiveServer, list[m.Lfm3Model]]) -> None:
    srv, models = onnx_server
    caps = httpx.get(srv.url + "/api/v1/info").json()["capabilities"]["realtime"]["pipelines"]["speech-to-speech"]
    assert caps["configured"] and caps["available"] and caps["model"] == "lfm3-audio-2b-onnx"
    ws = await websockets.connect(srv.ws_url + "/v1/realtime?model=gpt-realtime")
    try:
        assert json.loads(await ws.recv())["type"] == "session.created"
        session = {
            "type": "realtime", "pipeline": "speech-to-speech", "instructions": "Be brief.",
            "audio": {"input": {"transcription": {"model": "fake-stt"},
                                "turn_detection": {"type": "server_vad", "silence_duration_ms": 300}}},
        }  # fmt: skip
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        assert (await _until(ws, "session.updated"))[-1]["session"]["pipeline"] == "speech-to-speech"
        for chunk in b64_pcm24(2.0):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        events = await _until(ws, "response.done")
    finally:
        await ws.close()
    done = events[-1]["response"]
    assert done["status"] == "completed"
    assert done["output"][0]["content"][0]["transcript"] == "Hello there."
    audio = sum(len(e["delta"]) for e in events if e["type"] == "response.output_audio.delta") * 3 // 4
    assert audio == 14 * 1920 * 2  # 14 speech frames as 24 kHz PCM16
    assert done["usage"]["output_tokens"] == 12 + 14  # text tokens and speech frames
    graphs = models[0].graphs
    assert isinstance(graphs, TextGraphs)
    prompt = prompt_of(graphs)
    assert prompt.startswith("<|startoftext|><|im_start|>system\n" + m.SYSTEM_MIXED + "\n\nBe brief.<|im_end|>\n")
    heard = prompt.count("<|audio|>")
    assert 25 <= heard <= 36  # the turn's 2 s of speech (12.5 audio tokens per second) and its padding
    assert graphs.encode_calls >= 2  # prefilled while the user was speaking
    described = httpx.get(srv.url + "/api/v1/info").json()["capabilities"]["realtime"]["pipelines"]["speech-to-speech"]
    assert described["backend"] == "onnx" and described["in_process"] and described["loaded"]


async def test_backend_needs_packages_and_files(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings.s2s_model = "lfm3-audio-2b-onnx"
    core = Bolo(settings)
    monkeypatch.setattr(m, "missing_packages", lambda: ["onnxruntime", "tokenizers"])
    described = core.s2s.describe()
    assert described["configured"] and not described["available"]
    assert (
        "onnxruntime and tokenizers not installed" in described["reason"] and "bolo-speech[s2s]" in described["reason"]
    )
    with pytest.raises(ComponentMissing, match="needs onnxruntime and tokenizers") as missing:
        await core.s2s.backend()
    assert missing.value.status == 424
    monkeypatch.setattr(m, "missing_packages", lambda: [])
    described = core.s2s.describe()
    assert described["configured"] and not described["available"] and "is not installed" in described["reason"]
    with pytest.raises(Exception, match="not installed and auto_pull is off"):
        await core.s2s.backend()
    # "installed", but a file is gone: a clear 503 when the backend is built, not in the middle of a turn
    monkeypatch.setattr(core.store, "is_installed", lambda model_id: True)
    assert core.s2s.describe()["available"]

    def no_files(path: Path, options: dict[str, Any], threads: int, device: str = "cpu") -> m.Lfm3Model:
        raise EngineUnavailable(f"{path / 'tokenizer.json'} is missing; pull the model again")

    monkeypatch.setattr(m, "shared_model", no_files)
    with pytest.raises(EngineUnavailable, match="is missing; pull the model again") as gone:
        await core.s2s.backend()
    assert gone.value.status == 503
    await core.s2s.close()


def test_catalog_entry_and_engine(settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    core = Bolo(settings)
    spec = core.catalog.resolve("lfm3-audio-2b-onnx")
    assert spec is not None and spec.kind == m.KIND and spec.engine == "onnxruntime" and spec.task == "tts"
    assert spec.commercial_use is False and "LFM Open License" in spec.license
    assert [v["id"] for v in spec.voices] == ["female", "male"] and spec.options["precision"] == "q4"
    paths = {u["path"] for u in spec.source.urls}
    assert {"onnx/" + f for f in m.PRECISIONS["q4"]} <= paths and "onnx/depth_decoder.onnx" in paths
    assert {"tokenizer.json", "config.json", "generation_config.json", "preprocessor_config.json",
            "audio_tokenizer/config.json", "LICENSE"} <= paths  # fmt: skip
    assert all(len(u["sha256"]) == 64 and u["size"] > 0 for u in spec.source.urls)
    assert sum(u["size"] for u in spec.source.urls) == spec.size_bytes
    # the engine hands the directory to this module: one model for speech-to-speech and text-to-speech
    seen: list[tuple[Path, int]] = []

    def shared(path: Path, options: dict[str, Any], threads: int, device: str = "cpu") -> m.Lfm3Model:
        seen.append((path, threads))
        return fake_model()

    monkeypatch.setattr(m, "shared_model", shared)
    engine = core.engines.get("onnxruntime")
    assert engine is not None and m.KIND in engine.kinds
    voice = engine.load(m.KIND, "tts", tmp_path, {"voices": spec.voices, **spec.options}, LoadOptions(threads=3))
    assert isinstance(voice, m.Lfm3Tts) and seen == [(tmp_path, 3)] and voice.sample_rate == 24000
