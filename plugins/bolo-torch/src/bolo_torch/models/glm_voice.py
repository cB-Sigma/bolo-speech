"""Zhipu GLM-4-Voice 9B as a turn-based speech-to-speech model (text and speech interleaved, streamed).

The authors' recipe (github.com/THUDM/GLM-4-Voice at the pinned commit: ``web_demo.py``, ``model_server.py``,
``speech_tokenizer/utils.py``, ``flow_inference.py``):

* a user turn's 16 kHz speech becomes discrete tokens from the Whisper-based ``WhisperVQEncoder`` of
  ``glm-4-voice-tokenizer`` (30 s at a time), written as ``<|begin_of_audio|><|audio_N|>...<|end_of_audio|>``;
* the prompt is ``<|system|>\\n<instruction>`` then ``<|user|>\\n...`` and ``<|assistant|>streaming_transcription\\n``
  per turn; the system instruction asks for "13 text tokens followed by 26 audio tokens";
* the language model (``glm-4-voice-9b``: ChatGLM, Hugging Face remote code, opt in with ``allow_remote_code``)
  generates; ids from ``<|audio_0|>`` up are speech tokens, the rest text, ``<|user|>`` ends the answer;
* speech tokens go to the flow-matching decoder with HiFi-GAN (``glm-4-voice-decoder``, ``AudioDecoder.token2wav``)
  in blocks of 25, 50, 100, 150 and then 200 tokens, each block conditioned on what came before; 22.05 kHz.

Streaming: text as the model writes it; the first audio after 25 speech tokens (about 2 s of speech at 12.5 tokens a
second), which come after the first 13 text tokens. One voice. The model's earlier answers go back as their text
(bolo keeps no speech tokens of them), so the model sees its own past turns without their audio tokens.
``quantization: int4`` (bitsandbytes NF4, as the authors' ``model_server.py --dtype int4``) brings the language model
to about 6 GB. Licence: the model weights are under the glm-4-voice License (academic use free; commercial use only
after registering with Zhipu); the code is Apache-2.0.
"""

from __future__ import annotations

import importlib
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from bolo.engines.base import ConversationChunk, ConversationRequest, ConversationTurn, SpeechToSpeechModel

from bolo_torch import remote_code, upstream
from bolo_torch.models import peak_safe, to_voices
from bolo_torch.models.s2s_util import GrowingText, TokenQueue, generate_in_thread, no_grad, stopping_criteria, to_numpy
from bolo_torch.runtime import InvalidRequest, LoadContext, model_dir, quantized_device_map, require, split_kwargs

log = logging.getLogger("bolo_torch.glm_voice")

SAMPLE_RATE = 22050
AUDIO_SYSTEM = (
    "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and "
    "respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
)
TEXT_SYSTEM = (
    "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond "
    "in a interleaved manner, with 13 text token followed by 26 audio tokens."
)
BLOCKS = (25, 50, 100, 150, 200)
TOKENIZER_DIR = "glm-4-voice-tokenizer"
DECODER_DIR = "glm-4-voice-decoder"
LICENSE_NOTE = (
    "GLM-4-Voice's weights are under the glm-4-voice License: free for academic research; commercial use only after "
    "registering at https://open.bigmodel.cn/mla/form (then free); show 'Built with glm-4'; no military or illegal use; "
    "Chinese law applies."
)


def speech_tokens(model: Any, feature_extractor: Any, audio: np.ndarray, device: str) -> list[int]:
    """``speech_tokenizer.utils.extract_speech_token`` for one 16 kHz clip, on ``device`` (the original moves
    everything to CUDA): 30 s windows, padded to the encoder's stride, tokens where the attention mask is set."""
    import torch

    pieces = [audio[t * 16000 : (t + 30) * 16000] for t in range(0, max(1, -(-len(audio) // 16000)), 30)]
    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return []
    pooling = model.config.pooling_kernel_size or 1
    stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling * feature_extractor.hop_length
    tokens: list[int] = []
    with torch.no_grad():
        features = feature_extractor(
            pieces, sampling_rate=16000, return_attention_mask=True, return_tensors="pt", padding="longest",
            pad_to_multiple_of=stride,
        )  # fmt: skip
        features = features.to(device=device)
        outputs = model(**features)
        ids = outputs.quantized_token_ids
        mask = features["attention_mask"][:, :: model.conv1.stride[0] * model.conv2.stride[0]]
        mask = mask[:, :: model.config.pooling_kernel_size]
        for i in range(len(ids)):
            tokens.extend(int(t) for t in ids[i][mask[i].bool()].tolist())
    return tokens


class GlmVoiceS2S(SpeechToSpeechModel):
    sample_rate = SAMPLE_RATE
    supports_tools = False
    placement_report: dict[str, Any] | None = None

    def __init__(
        self,
        glm: Any,
        tokenizer: Any,
        whisper: Any,
        feature_extractor: Any,
        decoder: Any,
        *,
        device: str = "cuda",
        max_new_tokens: int = 2000,
        temperature: float = 0.2,
        top_p: float = 0.8,
    ) -> None:
        self.glm, self.tokenizer, self.whisper, self.features, self.decoder = (
            glm, tokenizer, whisper, feature_extractor, decoder,
        )  # fmt: skip
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature, self.top_p = temperature, top_p
        self.voices = to_voices([{"id": "default", "name": "GLM-4-Voice"}])
        self.audio_offset = int(tokenizer.convert_tokens_to_ids("<|audio_0|>"))
        self.end_token = int(tokenizer.convert_tokens_to_ids("<|user|>"))
        self._lock = threading.Lock()

    def prompt(self, request: ConversationRequest) -> str:
        """The demo's prompt for the whole conversation."""
        spoken_last = False
        parts: list[str] = []
        for turn in request.turns:
            if turn.role == "assistant":
                parts.append(f"<|assistant|>streaming_transcription\n{turn.text}")
                continue
            content, spoken_last = self._user(turn)
            parts.append(f"<|user|>\n{content}")
        system = AUDIO_SYSTEM if spoken_last else TEXT_SYSTEM
        if request.instructions and request.instructions.strip():
            system = system.rstrip() + "\n" + request.instructions.strip()
        return f"<|system|>\n{system}" + "".join(parts) + "<|assistant|>streaming_transcription\n"

    def _user(self, turn: ConversationTurn) -> tuple[str, bool]:
        if turn.role == "tool":
            return f"(result of a tool call: {turn.text})", False
        out = ""
        spoken = turn.audio is not None and len(turn.audio) > 0
        if spoken:
            ids = speech_tokens(self.whisper, self.features, np.asarray(turn.audio, dtype=np.float32), self.device)
            out = "<|begin_of_audio|>" + "".join(f"<|audio_{x}|>" for x in ids) + "<|end_of_audio|>"
        if turn.text:
            out = (out + "\n" + turn.text) if out else turn.text
        return out, spoken

    def voice(self, name: str | None) -> str:
        if name and name.lower() not in ("default", "glm-4-voice"):
            raise InvalidRequest(f"unknown voice {name!r}; GLM-4-Voice has one voice (default)")
        return "default"

    def respond(
        self, request: ConversationRequest, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[ConversationChunk]:
        if cancelled is not None and cancelled():
            return
        if request.audio_out:
            self.voice(request.voice)
        limit = min(request.max_output_tokens or self.max_new_tokens, self.max_new_tokens)
        with self._lock, no_grad():
            import torch

            inputs = self.tokenizer([self.prompt(request)], return_tensors="pt").to(self.device)
            stream = TokenQueue(skip_prompt=True)
            stop = threading.Event()

            def halted() -> bool:
                return stop.is_set() or (cancelled is not None and cancelled())

            kwargs = {
                **inputs, "max_new_tokens": limit, "temperature": self.temperature, "top_p": self.top_p,
                "stopping_criteria": stopping_criteria(halted),
            }  # fmt: skip
            thread = generate_in_thread(self.glm.generate, kwargs, stream)
            text = GrowingText(self._decode)
            audio_tokens: list[int] = []
            block = 0
            session = str(uuid.uuid4())
            prompt_token = torch.zeros(1, 0, dtype=torch.int64)
            mels: list[Any] = []
            try:
                finalize = False
                for token in stream:
                    if halted():
                        break
                    if token == self.end_token:
                        finalize = True
                    if request.audio_out and (len(audio_tokens) >= BLOCKS[block] or (finalize and audio_tokens)):
                        speech, prompt_token = self._decode_block(audio_tokens, session, prompt_token, mels, finalize)
                        block = min(block + 1, len(BLOCKS) - 1)
                        audio_tokens = []
                        if speech.size:
                            yield ConversationChunk(audio=speech)
                    if finalize:
                        break
                    if token >= self.audio_offset:
                        audio_tokens.append(token - self.audio_offset)
                    else:
                        piece = text.add(token)
                        if piece:
                            yield ConversationChunk(text=piece)
                if halted():
                    return
                rest = text.flush()
                if rest:
                    yield ConversationChunk(text=rest)
                if request.audio_out and audio_tokens:  # generation ended without <|user|>
                    speech, _ = self._decode_block(audio_tokens, session, prompt_token, mels, True)
                    if speech.size:
                        yield ConversationChunk(audio=speech)
            finally:
                stop.set()
                thread.join()

    def _decode_block(
        self, tokens: list[int], session: str, prompt_token: Any, mels: list[Any], finalize: bool
    ) -> tuple[np.ndarray, Any]:
        """One block through ``AudioDecoder.token2wav``, conditioned on the blocks before (the demo's loop)."""
        import torch

        tts_token = torch.tensor(tokens, device=self.device).unsqueeze(0)
        prompt_feat = torch.cat(mels, dim=-1).transpose(1, 2) if mels else torch.zeros(1, 0, 80)
        speech, mel = self.decoder.token2wav(
            tts_token, uuid=session, prompt_token=prompt_token.to(self.device), prompt_feat=prompt_feat.to(self.device),
            finalize=finalize,
        )  # fmt: skip
        mels.append(mel)
        return peak_safe(to_numpy(speech)), torch.cat((prompt_token.to(tts_token.device), tts_token), dim=-1)

    def _decode(self, ids: list[int]) -> str:
        return str(self.tokenizer.decode(ids, spaces_between_special_tokens=False))


def load(ctx: LoadContext) -> GlmVoiceS2S:
    path = model_dir(ctx.path, ctx.options)
    trust = remote_code.trust_remote_code(ctx, path)
    remote_code.check_pinned_files(path, ctx.options, [f"{DECODER_DIR}/config.yaml"], ctx.model_id)
    log.warning("%s: %s", ctx.model_id, ctx.option("license_note", LICENSE_NOTE))
    transformers = require("transformers", "glm-4-voice")
    code = upstream.ensure("glm-4-voice")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust)
    kwargs: dict[str, Any] = {"trust_remote_code": trust, "torch_dtype": ctx.dtype}
    quantization = str(ctx.option("quantization", "") or "").lower()
    if quantization in ("int4", "nf4"):
        import torch

        require("bitsandbytes", "glm-4-voice")
        # when bolo asks for a GPU/CPU split, the 4-bit model is split the same way (what does not fit stays on
        # the CPU, unquantized); otherwise it loads whole on the device
        offload = split_kwargs(ctx)
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=bool(offload),
        )  # fmt: skip
        if offload:
            # an explicit map, worked out here: with device_map="auto" transformers converts the layers bound for
            # the CPU to 4-bit too, and they cannot be loaded there (meta tensors)
            kwargs["device_map"] = quantized_device_map(
                ctx, offload, transformers.AutoModel, path, bits=4, trust_remote_code=trust
            )
        else:
            kwargs["device_map"] = {"": 0 if ctx.device == "cuda" else ctx.device}
    elif quantization:
        raise InvalidRequest(f"unknown quantization {quantization!r} (int4, or none)")
    split = {} if "device_map" in kwargs else split_kwargs(ctx, transformers.AutoModel, path, trust_remote_code=trust)
    glm = transformers.AutoModel.from_pretrained(str(path), **kwargs, **split)
    if "device_map" not in kwargs and not split:
        glm = glm.to(ctx.device)
    glm.eval()
    whisper_mod = importlib.import_module("speech_tokenizer.modeling_whisper")
    whisper = whisper_mod.WhisperVQEncoder.from_pretrained(str(path / TOKENIZER_DIR)).eval().to(ctx.device)
    features = transformers.WhisperFeatureExtractor.from_pretrained(str(path / TOKENIZER_DIR))
    flow = importlib.import_module("flow_inference")
    decoder_dir = path / DECODER_DIR
    decoder = flow.AudioDecoder(
        config_path=str(decoder_dir / "config.yaml"), flow_ckpt_path=str(decoder_dir / "flow.pt"),
        hift_ckpt_path=str(decoder_dir / "hift.pt"), device=ctx.device,
    )  # fmt: skip
    log.info("%s: GLM-4-Voice code from %s", ctx.model_id, code)
    model = GlmVoiceS2S(
        glm, tokenizer, whisper, features, decoder, device=ctx.device,
        max_new_tokens=int(ctx.option("max_new_tokens", 2000)), temperature=float(ctx.option("temperature", 0.2)),
        top_p=float(ctx.option("top_p", 0.8)),
    )  # fmt: skip
    if split:
        from bolo_torch.runtime import device_map_report

        model.placement_report = device_map_report(glm)
    return model
