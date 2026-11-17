"""Spoken-language identification (bolo task ``lid``) in PyTorch.

``voxlect-lid``: Voxlect's classifiers (Feng et al., 2025, github.com/tiantiaf0627/voxlect) of the language spoken in
India. A speech encoder with LoRA adapters in its upper layers, then point-wise 1-D convolutions, mean pooling over
time and a two-layer head. bolo-torch rebuilds them from transformers' own encoder classes and the checkpoint's
weights (the LoRA products are merged into the weights at load time), without the Voxlect code:

* ``pretrain_model: whisper_small``: Whisper small's encoder on 15 s of audio (log-mel features from a 15-second
  Whisper feature extractor, 750 positions); the last hidden state is pooled over the frames the audio fills;
* ``pretrain_model: mms-lid-256``: MMS-LID-256's wav2vec 2.0 encoder on normalised audio; the hidden states of all
  layers are mixed with the checkpoint's learnt softmax weights, then pooled over every frame.

The checkpoint's ``config.json`` names the encoder; the encoder's own ``config.json`` (from openai/whisper-small or
facebook/mms-lid-256) is downloaded by bolo into ``backbone/``. Options: ``labels`` (the class order, as language
codes), ``max_s`` (15). Extra ``voxlect`` (transformers, safetensors).

``hf-audio-classification``: any transformers audio classifier whose labels are languages (``config.json`` with
``id2label``, a ``preprocessor_config.json``; HuBERT, wav2vec 2.0, ...), loaded with transformers' Auto classes
from the downloaded folder (never remote code). Labels written as language names (``hindi``) or codes are turned
into codes. Options: ``max_s`` (30). Extra ``voxlect`` (transformers, safetensors).

``speechbrain-lid``: a SpeechBrain ECAPA-TDNN language classifier stored as ``embedding_model.ckpt``,
``classifier.ckpt`` and a label file, built from SpeechBrain's own classes without its hub loader (which would look
for the files in another repository). SpeechBrain's classifier answers a cosine score per language; bolo-torch turns
the scores into probabilities with a softmax at ``scale`` (30, the margin-softmax scale such models are trained
with). Options: ``n_mels`` (80), ``labels_file`` (``language_encoder.txt``), ``scale``, ``max_s`` (30).
Extra ``speechbrain``.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from bolo.engines.base import FloatArray, LidModel

from bolo_torch.runtime import InvalidRequest, LoadContext, model_dir, require

SR = 16000


def _ranked(labels: list[str], probs: Any) -> list[tuple[str, float]]:
    pairs = [(labels[i], float(p)) for i, p in enumerate(probs) if i < len(labels)]
    return sorted(pairs, key=lambda kv: -kv[1])


def merge_lora(state: dict[str, Any], scaling: float) -> dict[str, Any]:
    """Fold loralib's ``lora_A`` / ``lora_B`` pairs into their layer's ``weight`` (W + B A * scaling)."""
    out = dict(state)
    for key in [k for k in state if k.endswith(".lora_A")]:
        base = key[: -len(".lora_A")]
        a, b = out.pop(key), out.pop(base + ".lora_B")
        out[base + ".weight"] = out[base + ".weight"] + (b @ a) * scaling
    return out


def _head(torch: Any, size: int, hidden: int, classes: int) -> tuple[Any, Any]:
    nn = torch.nn
    seq = nn.Sequential(
        nn.Conv1d(size, hidden, 1), nn.ReLU(), nn.Dropout(0.1), nn.Conv1d(hidden, hidden, 1), nn.ReLU(),
        nn.Dropout(0.1), nn.Conv1d(hidden, hidden, 1),
    )  # fmt: skip
    out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, classes))
    return seq, out


def _sub(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


class VoxlectLid(LidModel):
    def __init__(
        self, kind: str, encoder: Any, seq: Any, out: Any, weights: Any, labels: list[str], device: str,
        max_s: float, extractor: Any,
    ) -> None:  # fmt: skip
        self.kind = kind  # "whisper" or "mms"
        self.encoder, self.seq, self.out, self.weights = encoder, seq, out, weights
        self.labels = labels
        self.device = device
        self.max_s = max_s
        self.extractor = extractor
        self._lock = threading.Lock()

    def identify(self, samples: FloatArray) -> list[tuple[str, float]]:
        import torch

        x = np.ascontiguousarray(samples[: int(self.max_s * SR)], dtype=np.float32)
        if len(x) < SR // 10:
            x = np.pad(x, (0, SR // 10 - len(x)))
        with self._lock, torch.inference_mode():
            if self.kind == "whisper":
                n = int(self.max_s * SR)
                feats = self.extractor(x, sampling_rate=SR, return_tensors="pt", max_length=n).input_features
                hidden = self.encoder(feats.to(self.device)).last_hidden_state
                frames = (len(x) // 160 - 1) // 2 + 1  # the encoder frames the audio fills (Voxlect's pooling)
                hidden = hidden[:, : max(1, frames)]
            else:
                norm = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
                states = self.encoder(torch.from_numpy(norm)[None].to(self.device), output_hidden_states=True)
                stacked = torch.stack(states.hidden_states, dim=0)
                mix = torch.softmax(self.weights, dim=-1).view(-1, 1, 1, 1)
                hidden = (mix * stacked).sum(dim=0)
            feats = self.seq(hidden.transpose(1, 2)).transpose(1, 2).mean(dim=1)
            probs = torch.softmax(self.out(feats)[0].float(), dim=-1).cpu().numpy()
        return _ranked(self.labels, probs)


def load_voxlect(ctx: LoadContext) -> VoxlectLid:
    require("transformers", "voxlect")
    st = require("safetensors.torch", "voxlect")
    import torch
    import transformers

    root = model_dir(ctx.path, ctx.options)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    backbone = root / "backbone"
    if not (backbone / "config.json").is_file():
        raise InvalidRequest(f"no encoder configuration in {backbone} (backbone/config.json)")
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise InvalidRequest(f"no Voxlect weights (*.safetensors) in {root}")
    labels = [str(x) for x in ctx.option("labels", [])]
    classes = int(cfg.get("output_class_num", len(labels)))
    if len(labels) != classes:
        raise InvalidRequest(f"options.labels has {len(labels)} languages, the model {classes}")
    state = merge_lora(st.load_file(str(files[0])), 1.0 / float(cfg.get("lora_rank", 16)))
    pretrain = str(cfg.get("pretrain_model", ""))
    if pretrain.startswith("whisper"):
        kind = "whisper"
        config = transformers.WhisperConfig.from_pretrained(str(backbone))
        config.max_source_positions = 750
        encoder = transformers.models.whisper.modeling_whisper.WhisperEncoder(config)
        encoder.load_state_dict(_sub(state, "backbone_model.encoder."), strict=True)
        max_s = float(ctx.option("max_s", 15.0))
        extractor = transformers.WhisperFeatureExtractor(feature_size=config.num_mel_bins, chunk_length=int(max_s))
        size = config.d_model
    elif pretrain.startswith("mms"):
        kind = "mms"
        config = transformers.Wav2Vec2Config.from_pretrained(str(backbone))
        encoder = transformers.Wav2Vec2Model(config)
        missing, unexpected = encoder.load_state_dict(_sub(state, "backbone_model."), strict=False)
        missing = [k for k in missing if "masked_spec_embed" not in k]
        if missing or unexpected:
            raise InvalidRequest(
                f"the Voxlect weights do not fit the encoder: missing {missing[:5]}, extra {unexpected[:5]}"
            )
        max_s = float(ctx.option("max_s", 15.0))
        extractor = None
        size = config.hidden_size
    else:
        raise InvalidRequest(f"bolo-torch cannot build a Voxlect model on {pretrain!r}")
    seq, out = _head(torch, size, int(cfg.get("hidden_dim", 256)), classes)
    seq.load_state_dict(_sub(state, "model_seq."))
    out.load_state_dict(_sub(state, "out_layer."))
    weights = state.get("weights", torch.zeros(1))
    for module in (encoder, seq, out):
        module.eval().to(ctx.device)
    return VoxlectLid(kind, encoder, seq, out, weights.to(ctx.device), labels, ctx.device, max_s, extractor)


class SpeechBrainLid(LidModel):
    def __init__(self, modules: dict[str, Any], labels: list[str], device: str, scale: float, max_s: float) -> None:
        self.m = modules
        self.labels = labels
        self.device = device
        self.scale = scale
        self.max_s = max_s
        self._lock = threading.Lock()

    def identify(self, samples: FloatArray) -> list[tuple[str, float]]:
        import torch

        x = np.ascontiguousarray(samples[: int(self.max_s * SR)], dtype=np.float32)
        if len(x) < SR // 10:
            x = np.pad(x, (0, SR // 10 - len(x)))
        with self._lock, torch.inference_mode():
            wav = torch.from_numpy(x)[None].to(self.device)
            lens = torch.ones(1, device=self.device)
            feats = self.m["features"](wav)
            feats = self.m["norm"](feats, lens)
            emb = self.m["embedding"](feats, lens)
            scores = self.m["classifier"](emb).reshape(-1).float()
            probs = torch.softmax(scores * self.scale, dim=-1).cpu().numpy()
        return _ranked(self.labels, probs)


#: language names as classifiers' label files write them -> the codes bolo's catalog uses
NAMES = {
    "english": "en", "hindi": "hi", "marathi": "mr", "tamil": "ta", "telugu": "te", "bengali": "bn",
    "kannada": "kn", "malayalam": "ml", "gujarati": "gu", "punjabi": "pa", "odia": "or", "oriya": "or", "urdu": "ur",
    "assamese": "as", "nepali": "ne", "sanskrit": "sa", "sindhi": "sd", "kashmiri": "ks", "konkani": "kok",
    "maithili": "mai", "manipuri": "mni", "meitei": "mni", "bodo": "brx", "dogri": "doi", "santali": "sat",
    "bhojpuri": "bho", "sinhala": "si",
}  # fmt: skip


def label_code(label: str) -> str:
    """A label (``hindi``, ``Hindi``, ``hi``, ``hin``) as the catalog's language code, where bolo-torch knows it."""
    key = label.strip().lower()
    return NAMES.get(key, key)


class HFAudioLid(LidModel):
    def __init__(self, model: Any, extractor: Any, labels: list[str], device: str, max_s: float) -> None:
        self.model, self.extractor = model, extractor
        self.labels = labels
        self.device = device
        self.max_s = max_s
        self._lock = threading.Lock()

    def identify(self, samples: FloatArray) -> list[tuple[str, float]]:
        import torch

        x = np.ascontiguousarray(samples[: int(self.max_s * SR)], dtype=np.float32)
        if len(x) < SR // 10:
            x = np.pad(x, (0, SR // 10 - len(x)))
        inputs = self.extractor(x, sampling_rate=SR, return_tensors="pt")
        with self._lock, torch.inference_mode():
            logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits[0].float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return _ranked(self.labels, probs)


def load_hf_classifier(ctx: LoadContext) -> HFAudioLid:
    transformers = require("transformers", "voxlect")
    root = model_dir(ctx.path, ctx.options)
    extractor = transformers.AutoFeatureExtractor.from_pretrained(str(root))
    model = transformers.AutoModelForAudioClassification.from_pretrained(str(root))
    model.eval().to(ctx.device)
    id2label = model.config.id2label
    labels = [label_code(str(id2label[i])) for i in range(len(id2label))]
    return HFAudioLid(model, extractor, labels, ctx.device, float(ctx.option("max_s", 30.0)))


def read_label_encoder(path: Path) -> list[str]:
    """SpeechBrain's CategoricalEncoder file (``'hindi' => 1``) as language codes in index order."""
    found: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("="):  # the encoder's own settings follow ('starting_index' => 0)
            break
        m = re.match(r"^'(.+)' => (\d+)$", line.strip())
        if m:
            label = m.group(1)
            found[int(m.group(2))] = label_code(label)
    return [found[i] for i in sorted(found)]


def load_speechbrain(ctx: LoadContext) -> SpeechBrainLid:
    require("speechbrain", "speechbrain")
    import torch
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier
    from speechbrain.processing.features import InputNormalization

    root = model_dir(ctx.path, ctx.options)
    labels = read_label_encoder(root / str(ctx.option("labels_file", "language_encoder.txt")))
    if not labels:
        raise InvalidRequest(f"no labels in {root / str(ctx.option('labels_file', 'language_encoder.txt'))}")
    n_mels = int(ctx.option("n_mels", 80))
    embedding = ECAPA_TDNN(
        input_size=n_mels, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192,
    )  # fmt: skip
    classifier = Classifier(input_size=192, out_neurons=len(labels))
    for module, name in ((embedding, "embedding_model.ckpt"), (classifier, "classifier.ckpt")):
        module.load_state_dict(torch.load(str(root / name), map_location="cpu", weights_only=True))
        module.eval().to(ctx.device)
    modules = {
        "features": Fbank(n_mels=n_mels).to(ctx.device),
        "norm": InputNormalization(norm_type="sentence", std_norm=False).to(ctx.device),
        "embedding": embedding,
        "classifier": classifier,
    }
    return SpeechBrainLid(
        modules, labels, ctx.device, float(ctx.option("scale", 30.0)), float(ctx.option("max_s", 30.0))
    )
