"""DiCoW in ONNX Runtime (bolo.engines.dicow_onnx) with a scripted fake network: STNO masks, Whisper's
timestamp rules, segments and seeking, the long-form loop, token decoding and the request path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bolo.config import Settings
from bolo.core import Bolo
from bolo.custom import choose_files, detect_kind, register_local_model
from bolo.diarize import DiarizeOptions, conditioning_activity
from bolo.engines import dicow_onnx as d
from bolo.engines.base import LoadOptions, SpeakerActivity, SpeakerRequest, SpeakerTurn, STTRequest
from bolo.store import RemoteFile

SR = 16000
GEN = d.GenerationSettings(suppress=[1, 2, 50358], lang_to_id={"en": 50259, "de": 50261})
TS = GEN.timestamp_begin  # <|0.00|>
EOS = GEN.eos
SOT = GEN.start


def ts(seconds: float) -> int:
    return TS + round(seconds / d.TIME_PRECISION)


# -- masks --------------------------------------------------------------------------------------------


def reference_stno(diar_mask: np.ndarray, s_index: int) -> np.ndarray:
    """``DiCoWPipeline.get_stno_mask`` (BUTSpeechFIT/DiCoW ``pipeline.py``, Apache-2.0), in NumPy."""
    non_target = np.ones(diar_mask.shape[0], bool)
    non_target[s_index] = False
    sil_frames = (1 - diar_mask).prod(axis=0)
    anyone_else = (1 - diar_mask[non_target]).prod(axis=0)
    target_spk = diar_mask[s_index] * anyone_else
    non_target_spk = (1 - diar_mask[s_index]) * (1 - anyone_else)
    overlapping_speech = diar_mask[s_index] - target_spk
    return np.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0)


@pytest.mark.parametrize("speakers", [1, 2, 3, 5])
def test_stno_masks_match_the_reference_formula(speakers: int) -> None:
    rng = np.random.default_rng(speakers)
    for activity in (
        (rng.random((speakers, 400)) > 0.6).astype(np.float32),
        rng.random((speakers, 400)).astype(np.float32),
    ):
        ours = d.stno_masks(activity)
        ref = np.stack([reference_stno(activity.astype(np.float64), s) for s in range(speakers)])
        np.testing.assert_allclose(ours, ref, atol=1e-6)
        np.testing.assert_allclose(ours.sum(axis=1), 1.0, atol=1e-6)


def test_turns_give_the_reference_masks_on_its_20ms_grid() -> None:
    turns = [SpeakerTurn(0.013, 1.23, 0), SpeakerTurn(1.01, 4.47, 1), SpeakerTurn(2.07, 3.011, 0)]
    activity, _ = conditioning_activity(turns, None, 5.0, "turns")
    ours = d.stno_masks(d.activity_at_mask_rate(activity.probs, activity.frame_s, 1500))
    diar = np.zeros((2, 1500))
    for t in turns:  # the reference's get_diarization_mask: round(t * 50)
        diar[t.speaker, round(t.start * 50) : round(t.end * 50)] = 1
    ref = np.stack([reference_stno(diar, k) for k in range(2)])
    np.testing.assert_array_equal(ours, ref)
    assert d.mask_frames(10 * SR) == 1500 and d.mask_frames(45 * SR) == 2250
    with pytest.raises(ValueError):
        d.activity_at_mask_rate(np.zeros(5, np.float32), 0.01, 10)


def test_windows_pad_features_with_zeros_and_masks_with_silence() -> None:
    features = np.ones((128, 4500), np.float32)
    masks = np.zeros((4, 2250), np.float32)
    masks[1] = 1.0
    feats, mask, num = d.window_inputs(features, masks, 2000, 4500)
    assert num == 2500 and feats[:, :2500].all() and not feats[:, 2500:].any()
    assert mask[1, :1250].all() and not mask[1, 1250:].any() and mask[0, 1250:].all() and not mask[0, :1250].any()
    feats, mask, num = d.window_inputs(features, masks, 0, 4500)
    assert num == 3000 and feats.all() and mask[1].all()


def test_log_mel_pads_short_input_to_30_s() -> None:
    assert d.log_mel(np.zeros(SR, np.float32)).shape == (128, 3000)
    assert d.log_mel(np.zeros(40 * SR + 7, np.float32)).shape == (128, 4000)


# -- tokens and timestamp rules -------------------------------------------------------------------------


def test_token_decoding_skips_special_tokens_and_cleans_up_spaces() -> None:
    vocab = {"Ġit": 1, "'s": 2, "Ġcaf": 3, "Ã©": 4, "Ġok": 5, "Ġ.": 6, "<|endoftext|>": EOS}
    tokens = d.TokenDecoder(vocab, EOS)
    assert tokens.decode([ts(0.0), 1, 2, 3, 4, ts(1.0)]) == " it's café"
    assert tokens.decode([5, 6, EOS, SOT]) == " ok."
    assert d.TokenDecoder(vocab, EOS, clean_up=False).decode([5, 6]) == " ok ."


def test_timestamp_rules() -> None:
    vocab = TS + 1500
    scores = np.zeros((1, vocab), np.float32)
    scores[0, 100] = 5.0
    scores[0, EOS] = 3.0
    # first token: a timestamp, or end-of-text (DiCoW's exception) at its own score
    out = d.apply_timestamp_rules(scores, [[]], GEN)
    assert np.isneginf(out[0, 100]) and out[0, EOS] == 3.0 and out[0, TS] == 0.0
    # an opening timestamp is followed by text; <|notimestamps|> never comes
    out = d.apply_timestamp_rules(scores, [[ts(1.0)]], GEN)
    assert out[0, 100] == 5.0 and np.isneginf(out[0, GEN.no_timestamps]) and np.isneginf(out[0, TS:]).all()
    # after text, no timestamp earlier than the last one
    out = d.apply_timestamp_rules(scores, [[ts(1.0), 100]], GEN)
    assert np.isneginf(out[0, ts(0.5)]) and out[0, ts(1.0) + 1] == 0.0
    # after text and a timestamp: no text (a timestamp closing the pair, or end-of-text); not earlier than it
    unlikely = scores.copy()
    unlikely[0, TS:] = -10.0
    out = d.apply_timestamp_rules(unlikely, [[ts(1.0), 100, ts(2.0)]], GEN)
    assert np.isneginf(out[0, 100]) and out[0, EOS] == 3.0 and out[0, ts(2.0)] == -10.0
    assert np.isneginf(out[0, ts(1.9)])
    # after a pair of timestamps: text only
    out = d.apply_timestamp_rules(scores, [[ts(1.0), 100, ts(2.0), ts(2.0)]], GEN)
    assert out[0, 100] == 5.0 and np.isneginf(out[0, TS:]).all()
    # when the timestamps together are more likely than any text token, a timestamp is forced
    many = np.full((1, vocab), 0.0, np.float32)
    many[0, 100] = 2.0
    out = d.apply_timestamp_rules(many, [[ts(1.0), 100]], GEN)
    assert np.isneginf(out[0, 100])


def test_segments_and_seek() -> None:
    # two closed segments and a single timestamp at the end: no speech after it, the whole window is consumed
    seq = [ts(0.0), 10, 11, ts(2.0), ts(2.5), 12, ts(4.0)]
    segs, offset = d.retrieve_segments(seq, 30.0, 3000, TS)
    assert [(round(s.start, 2), round(s.end, 2)) for s in segs] == [(30.0, 32.0), (32.5, 34.0)] and offset == 3000
    # the last segment is unfinished (cut by the window): drop it and seek to the last closed timestamp
    seq = [ts(0.0), 10, ts(2.0), ts(2.0), 11, ts(28.0), ts(28.0), 12, 13]
    segs, offset = d.retrieve_segments(seq, 0.0, 3000, TS)
    assert [(s.start, s.end) for s in segs] == [(0.0, 2.0), (2.0, 28.0)] and offset == 2800
    # no pair of timestamps: one segment between the last two timestamps
    segs, offset = d.retrieve_segments([ts(1.0), 10, 11, ts(3.0)], 0.0, 3000, TS)
    assert [(s.start, s.end) for s in segs] == [(1.0, 3.0)] and offset == 3000
    # a lone timestamp past 4 s with nothing after it: roll back to just before it (DiCoW's rule)
    segs, offset = d.retrieve_segments([ts(10.0)], 0.0, 3000, TS)
    assert segs == [] and offset == 900
    segs, offset = d.retrieve_segments([], 0.0, 1200, TS)
    assert segs == [] and offset == 1200


# -- the decode loop with a scripted network ------------------------------------------------------------


class ScriptedNetwork:
    """Emits a fixed token sequence per (encoder call, row of that call).

    The self-attention cache carries the tokens fed so far, so the step is known from the cache length."""

    def __init__(self, scripts: dict[tuple[int, int], list[int]], language: int = 50259) -> None:
        self.scripts = scripts
        self.language = language
        self.masks: list[np.ndarray] = []  # every encoded window's mask, in order
        self.vocab = TS + 1501

    def encode(self, features: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert features.shape[1:] == (128, 3000) and masks.shape[1:] == (4, 1500)
        call = len({int(c) for c in getattr(self, "_calls", [])})
        self._calls = [*getattr(self, "_calls", []), call]
        self.masks.extend(masks)
        k = np.full((1, len(masks), 1, 1, 1), call, np.float32)
        v = np.arange(len(masks), dtype=np.float32).reshape(1, -1, 1, 1, 1)
        return k, v

    def empty_cache(self, batch: int) -> np.ndarray:
        return np.zeros((1, batch, 1, 0, 1), np.float32)

    def decode(
        self, ids: np.ndarray, self_k: np.ndarray, self_v: np.ndarray, cross_k: np.ndarray, cross_v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = np.concatenate([self_k, ids.astype(np.float32)[None, :, None, :, None]], axis=3)
        logits = np.zeros((len(ids), self.vocab), np.float32)
        for row in range(len(ids)):
            if k.shape[3] == 1 and ids[row, 0] == SOT:  # language detection
                logits[row, self.language] = 10.0
                continue
            script = self.scripts.get((int(cross_k[0, row, 0, 0, 0]), int(cross_v[0, row, 0, 0, 0])), [EOS])
            step = k.shape[3] - 3
            logits[row, script[step] if step < len(script) else EOS] = 20.0
        return logits, k, k


def talking_masks(n_spk: int, frames: int) -> np.ndarray:
    masks = np.zeros((n_spk, 4, frames), np.float32)
    masks[:, 1] = 1.0
    return masks


def test_long_form_decoding_moves_through_the_recording_per_speaker() -> None:
    scripts = {
        # speaker 0: two segments, the second cut by the window end -> seek to 20 s, then one more window
        (0, 0): [ts(0.0), 10, ts(5.0), ts(5.0), 11, ts(20.0), ts(20.0), 12],
        (1, 0): [ts(1.0), 13, ts(3.0)],
        # speaker 1: says nothing (end-of-text first): the whole window is consumed, then the rest
        (0, 1): [EOS],
        (1, 1): [EOS],
    }
    network = ScriptedNetwork(scripts)
    features = np.zeros((128, 4500), np.float32)
    segments, langs = d.decode_long_form(network, features, talking_masks(2, 2250), GEN, "en", "transcribe")
    assert [(s.start, s.end, s.tokens) for s in segments[0]] == [
        (0.0, 5.0, [ts(0.0), 10, ts(5.0)]),
        (5.0, 20.0, [ts(5.0), 11, ts(20.0), ts(20.0)]),  # an unfinished window keeps the pair
        (21.0, 23.0, [ts(1.0), 13, ts(3.0)]),
    ]
    assert segments[1] == [] and langs == ["en", "en"]
    # second windows: speaker 0 from 20 s (25 s of audio left), speaker 1 from 30 s: padded with silence
    second0, second1 = network.masks[2], network.masks[3]
    assert second0[1, :1250].all() and second0[0, 1250:].all() and not second0[1, 1250:].any()
    assert second1[1, :750].all() and second1[0, 750:].all()


def test_language_detection_and_max_length() -> None:
    long_script = [ts(0.0)] + [100] * 500
    network = ScriptedNetwork({(0, 0): long_script}, language=50261)
    features = np.zeros((128, 3000), np.float32)
    segments, langs = d.decode_long_form(network, features, talking_masks(1, 1500), GEN, None, "transcribe")
    assert langs == ["de"]
    # 448 tokens at most including the three prompt tokens; no closing timestamp -> one segment over the window
    assert len(segments[0][0].tokens) == GEN.max_length - 3 and segments[0][0].end == 30.0


def model_with(scripts: dict[tuple[int, int], list[int]]) -> d.DiCoWOnnx:
    vocab = {"Ġhello": 10, "Ġthere": 11, "Ġagain": 12, "Ġbye": 13, "<|endoftext|>": EOS}
    return d.DiCoWOnnx(ScriptedNetwork(scripts), GEN, d.TokenDecoder(vocab, EOS), batch=1)


def test_transcribe_speakers_returns_one_result_per_speaker_and_skips_silent_ones() -> None:
    model = model_with({(0, 0): [ts(0.0), 10, ts(1.0), ts(1.5), 11, ts(2.0)]})
    assert model.speaker_conditioned and model.sample_rate == SR and "en" in model.languages
    probs = np.zeros((500, 2), np.float32)  # 5 s at 10 ms; speaker 1 talks for 0.1 s only
    probs[:200, 0] = 1.0
    probs[300:310, 1] = 1.0
    audio = np.random.default_rng(0).standard_normal(5 * SR).astype(np.float32) * 0.01
    results = model.transcribe_speakers(SpeakerRequest(audio, SpeakerActivity(probs), "en"))
    assert [r.text for r in results] == ["hello there", ""]
    assert [(s.start, s.end, s.text) for s in results[0].segments] == [(0.0, 1.0, "hello"), (1.5, 2.0, "there")]
    # plain recognition: the whole input is the target speaker
    plain = model_with({(0, 0): [ts(0.0), 13, ts(2.0)]})
    out = plain.transcribe([STTRequest(audio)])
    assert out[0].text == "bye" and out[0].language == "en"


# -- catalog kind and the request path ------------------------------------------------------------------


def write_model_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    gen = {
        "eos_token_id": EOS, "decoder_start_token_id": SOT, "no_timestamps_token_id": GEN.no_timestamps,
        "max_length": 448, "suppress_tokens": [1, 2], "lang_to_id": {"<|en|>": 50259}, "task_to_id": GEN.task_to_id,
        "ctc_weight": 0,
    }  # fmt: skip
    (path / "generation_config.json").write_text(json.dumps(gen))
    (path / "config.json").write_text(json.dumps({"vocab_size": 51866, "max_target_positions": 448}))
    (path / "vocab.json").write_text(json.dumps({"Ġhello": 10, "Ġthere": 11, "<|endoftext|>": EOS}))
    (path / "encoder.int8.onnx").write_bytes(b"x")
    (path / "decoder.int8.onnx").write_bytes(b"x")
    (path / "dicow_onnx.json").write_text("{}")
    return path


def test_kind_detection_and_file_choice() -> None:
    names = ["encoder.int8.onnx", "decoder.int8.onnx", "dicow_onnx.json", "vocab.json"]
    assert detect_kind("codebucketsolutions/dicow-v3.3-onnx", names) == "dicow_onnx"
    files = [RemoteFile(n, 1) for n in [*names, "encoder.onnx", "encoder.onnx.data"]]
    assert "encoder.onnx" in choose_files("dicow_onnx", files) and "*.int8.onnx" not in choose_files(
        "dicow_onnx", files
    )
    assert "*.int8.onnx" in choose_files("dicow_onnx", files, precision="float")


def test_settings_from_the_model_directory(tmp_path: Path) -> None:
    path = write_model_dir(tmp_path / "m")
    gen = d.GenerationSettings.from_dir(path)
    assert gen.lang_to_id == {"en": 50259} and gen.suppress == [1, 2] and gen.timestamp_begin == 50365
    data = json.loads((path / "generation_config.json").read_text(encoding="utf-8"))
    data["ctc_weight"] = 0.3
    (path / "generation_config.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="CTC"):
        d.GenerationSettings.from_dir(path)


async def test_diarized_request_through_the_core(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bolo.engines.base import EngineStatus
    from bolo.engines.onnx_turn import OnnxRuntimeEngine

    created: list[ScriptedNetwork] = []

    def fake_network(encoder: Path, decoder: Path, threads: int, device: str = "cpu") -> ScriptedNetwork:
        assert encoder.name == "encoder.int8.onnx" and decoder.name == "decoder.int8.onnx"
        # the fake diarizer finds two speakers; each says its own word in its own time
        net = ScriptedNetwork({(0, 0): [ts(0.5), 10, ts(3.0)], (0, 1): [ts(5.0), 11, ts(8.0)]})
        created.append(net)
        return net

    monkeypatch.setattr(d, "OnnxNetwork", fake_network)
    monkeypatch.setattr(OnnxRuntimeEngine, "status", lambda self: EngineStatus(self.id, True, "test", None))
    settings.default_diarization = "fake-diarization-overlap"
    settings.probe_models = False  # the fake network exists in this process only
    core = Bolo(settings)
    await core.start()
    try:
        spec = register_local_model(core, str(write_model_dir(tmp_path / "dicow-onnx")))
        # registered with what the catalog entry says: conditioned on speakers, Nemotron as its diarizer
        assert spec.kind == "dicow_onnx" and spec.engine == "onnxruntime" and spec.languages == ["en"]
        assert "speaker-conditioned" in spec.features and spec.options["diarization_model"] == "nemotron-3-diarization"
        core.store.pull(spec.id, spec.source)
        audio = np.random.default_rng(1).standard_normal(int(10 * SR)).astype(np.float32) * 0.05
        t = await core.transcribe(
            audio, SR, model=spec.id, language="en", diarize=DiarizeOptions(model="fake-diarization-overlap")
        )
        assert len(created) == 1
        by_speaker = {s.speaker: s.text for s in t.segments}
        assert sorted(by_speaker.values()) == ["hello", "there"]
        assert core.traces.recent(1)[0].extra["conditioning"] == "turns"
    finally:
        await core.stop()


def test_load_options_reach_the_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_model_dir(tmp_path / "m")
    (path / "encoder.onnx").write_bytes(b"x")
    seen: list[tuple[str, str, int]] = []
    monkeypatch.setattr(
        d, "OnnxNetwork", lambda e, dec, t, device="cpu": seen.append((e.name, dec.name, t)) or ScriptedNetwork({})
    )
    model = d.DiCoWOnnx.from_dir(path, {"encoder": "encoder.onnx", "batch": 2}, LoadOptions(threads=3))
    assert seen == [("encoder.onnx", "decoder.int8.onnx", 3)] and model.batch == 2
    with pytest.raises(FileNotFoundError):
        d.DiCoWOnnx.from_dir(tmp_path, {}, LoadOptions())


# -- int8 products on x86 CPUs without VNNI --------------------------------------------------------------


def test_sessions_ask_for_exact_int8_products_on_cpus_without_vnni() -> None:
    pytest.importorskip("onnxruntime")
    options = d.session_options(3)
    assert options.get_session_config_entry(d.X64_QUANT_PRECISION) == "1"
    assert options.intra_op_num_threads == 3 and options.inter_op_num_threads == 1


def test_onnx_network_builds_both_sessions_with_those_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ort = pytest.importorskip("onnxruntime")
    seen: list[str] = []

    class Input:
        def __init__(self, name: str, shape: list[Any]) -> None:
            self.name, self.shape = name, shape

    class Session:
        def __init__(self, path: str, sess_options: Any, providers: Any) -> None:
            seen.append(sess_options.get_session_config_entry(d.X64_QUANT_PRECISION))

        def get_inputs(self) -> list[Input]:
            cache = [4, "batch", 20, "length", 64]
            names = ("input_ids", "self_k", "self_v", "cross_k", "cross_v")
            return [Input(n, cache if n == "self_k" else []) for n in names]

    monkeypatch.setattr(ort, "InferenceSession", Session)
    net = d.OnnxNetwork(tmp_path / "encoder.int8.onnx", tmp_path / "decoder.int8.onnx", 2)
    assert seen == ["1", "1"] and (net.layers, net.heads, net.head_dim) == (4, 20, 64)


def test_dynamically_quantized_products_do_not_saturate() -> None:
    """The int8 graphs' pattern (DynamicQuantizeLinear, MatMulInteger with per-channel int8 weights) with
    activations that quantize near 255 and full-range weights: with uint8 x int8 pairs added in 16 bits (AVX2
    without VNNI) about a third of the products overflow. With the engine's options the result is the exact
    product of the quantized operands on every CPU."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    k, n = 512, 64
    rng = np.random.default_rng(0)
    w = rng.normal(size=(k, n)).astype(np.float32)
    w_scale = (np.abs(w).max(axis=0) / 127).astype(np.float32)
    w_q = np.round(w / w_scale).astype(np.int8)
    nodes = [
        helper.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"]),
        helper.make_node("MatMulInteger", ["xq", "w", "xz", "wz"], ["y32"]),
        helper.make_node("Cast", ["y32"], ["yf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["xs", "ws"], ["s"]),
        helper.make_node("Mul", ["yf", "s"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "int8",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [8, n])],
        [
            numpy_helper.from_array(w_q, "w"),
            numpy_helper.from_array(np.zeros(n, np.int8), "wz"),
            numpy_helper.from_array(w_scale, "ws"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    x = (rng.normal(size=(8, k)) + 3.0).astype(np.float32)  # mostly positive: quantized activations near 255
    session = ort.InferenceSession(model.SerializeToString(), d.session_options(1), providers=["CPUExecutionProvider"])
    (y,) = session.run(None, {"x": x})
    lo, hi = min(0.0, float(x.min())), max(0.0, float(x.max()))
    x_scale = np.float32((hi - lo) / 255)
    x_zero = np.clip(np.round(-lo / x_scale), 0, 255)
    x_q = np.clip(np.round(x / x_scale) + x_zero, 0, 255)
    exact = ((x_q - x_zero) @ w_q.astype(np.float64)) * (np.float64(x_scale) * w_scale)
    np.testing.assert_allclose(y, exact, rtol=1e-4, atol=1e-3 * float(np.abs(exact).max()))
