"""Audio classification against the caller's own classes (task ``audio-classify``):

* **zero-shot**, with a CLAP model (contrastive language-audio pretraining: :class:`bolo.engines.base.AudioTextModel`):
  the caller writes the labels in plain words ("angry voice", "hold music", "a dog barking"); each label is put
  into a sentence (the prompt, "This is a sound of {}."), the text tower embeds it, and the recording's audio
  embedding is compared with each: ``similarity`` is the cosine, ``score`` the softmax over the labels of the
  similarities times the model's logit scale. Label embeddings are cached by the model.
* **trained classifiers**: a small head (multinomial logistic regression, or a one-hidden-layer MLP) trained here, in
  seconds on the CPU, on frozen embeddings of labelled example clips from an encoder bolo has: a CLAP audio tower,
  a speaker-embedding model, or an audio tagger (its 527 class probabilities as features). A trained classifier is
  saved as a local model (``<models>/<id>/classifier.json`` and ``head.npz``, a custom catalog entry of kind
  ``classifier_head``), listed by ``bolo list`` and served by the same endpoint.

A recording longer than what the encoder hears at once (CLAP: 10 s, 7 s for MS-CLAP) is heard in pieces whose
embeddings are averaged; with ``window`` (and ``hop``) it is also classified window by window.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.audio.resample import resample
from bolo.engines.base import AudioTextModel, FloatArray
from bolo.errors import BoloError, InvalidRequest

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo
    from bolo.obs.trace import Trace

TASK = "audio-classify"
#: the catalog kind of a trained classifier
HEAD_KIND = "classifier_head"
#: tasks whose models can be the encoder of a trained classifier
ENCODER_TASKS = ("audio-classify", "speaker-embedding", "audio-tagging")
RATE = 16000
MAX_LABELS = 100
MAX_LABEL_CHARS = 200
MAX_WINDOWS = 2000
#: a long recording is heard in pieces this long by encoders without a clip length of their own
PIECE_S = 10.0
#: examples per class below which training warns that there is too little data
FEW_EXAMPLES = 10
MAX_EXAMPLES = 2000
#: training examples embedded at once
EMBED_AT_ONCE = 8
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ClassifierNotFound(BoloError):
    status = 404
    code = "classifier_not_found"
    type = "invalid_request_error"


class ClassifierExists(BoloError):
    status = 409
    code = "classifier_exists"
    type = "invalid_request_error"


@dataclass
class Window:
    start: float
    end: float
    scores: list[tuple[str, float, float | None]]


@dataclass
class Classification:
    model: str
    duration: float
    #: (label, score, similarity); similarity is None for a trained classifier
    scores: list[tuple[str, float, float | None]]
    classifier: str | None = None
    prompt: str | None = None
    windows: list[Window] = field(default_factory=list)


# -- request parsing ---------------------------------------------------------------------------------------------
def parse_labels(values: list[Any]) -> list[str]:
    """Labels from request fields: a JSON array, or text with one label per line or comma-separated labels (a
    field may be repeated)."""
    out: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.startswith("["):
            try:
                items = json.loads(text)
            except ValueError as exc:
                raise InvalidRequest("labels: not a JSON array of strings", param="labels") from exc
            if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
                raise InvalidRequest("labels: not a JSON array of strings", param="labels")
            out.extend(items)
        else:
            out.extend(re.split(r"[\n,]", text) if ("\n" in text or "," in text) else [text])
    labels = list(dict.fromkeys(x.strip() for x in out if x.strip()))
    if len(labels) > MAX_LABELS:
        raise InvalidRequest(f"at most {MAX_LABELS} labels", param="labels")
    if any(len(x) > MAX_LABEL_CHARS for x in labels):
        raise InvalidRequest(f"a label is longer than {MAX_LABEL_CHARS} characters", param="labels")
    return labels


def prompt_text(prompt: str, label: str) -> str:
    """``label`` put into ``prompt`` (``{}`` or ``{label}`` marks where); a prompt with neither: the label alone."""
    if "{label}" in prompt:
        return prompt.replace("{label}", label)
    if "{}" in prompt:
        return prompt.replace("{}", label)
    return label


def softmax(z: FloatArray) -> FloatArray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    out: FloatArray = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    return out


def _unit(v: FloatArray) -> FloatArray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-12 else v.astype(np.float32)


def _spans(n: int, rate: int, window_s: float, hop_s: float) -> list[tuple[int, int]]:
    size, hop = int(window_s * rate), max(1, int(hop_s * rate))
    if n <= size:
        return [(0, n)]
    out = [(a, min(n, a + size)) for a in range(0, n - size // 4, hop)]
    if len(out) > MAX_WINDOWS:
        raise InvalidRequest(
            f"{len(out)} windows: at most {MAX_WINDOWS} per request; use a longer window or hop", param="window"
        )
    return out


def _pieces(a: int, b: int, size: int) -> list[tuple[int, int]]:
    """``[a, b)`` cut into pieces of at most ``size`` samples; a short last piece joins the one before."""
    if b - a <= size:
        return [(a, b)]
    out = [(s, min(b, s + size)) for s in range(a, b, size)]
    if len(out) > 1 and out[-1][1] - out[-1][0] < size // 4:
        out[-2] = (out[-2][0], out[-1][1])
        out.pop()
    return out


# -- embeddings from any encoder ---------------------------------------------------------------------------------
class Encoder:
    """Embeddings of stretches of a recording from a loaded encoder (a model entry of the manager)."""

    def __init__(self, entry: Any) -> None:
        self.entry = entry
        self.model = entry.model
        q = entry.queue
        if isinstance(self.model, AudioTextModel):
            self.how, self.rate, self.clip_s = "clap", int(self.model.sample_rate), float(self.model.clip_s)
        elif q.handles("embed"):
            self.how, self.rate, self.clip_s = "embed", RATE, 30.0
        elif q.handles("classify"):
            self.how, self.rate, self.clip_s = "tags", RATE, PIECE_S
        else:
            raise InvalidRequest(
                f"model '{entry.spec.id}' gives no audio embeddings: use a CLAP ({TASK}), speaker-embedding or "
                "audio-tagging model as the encoder",
                param="encoder",
            )
        self._labels: list[str] | None = list(getattr(self.model, "labels", None) or []) or None

    async def _one(self, piece: FloatArray) -> FloatArray:
        q = self.entry.queue
        if self.how == "clap":
            vector, _ = await q.submit("embed_audio", piece)
        elif self.how == "embed":
            vector, _ = await q.submit("embed", piece)
        else:
            scores, _ = await q.submit("classify", piece)
            table = {str(label): float(score) for label, score in scores}
            if self._labels is None:
                self._labels = sorted(table)
            vector = np.array([table.get(label, 0.0) for label in self._labels], dtype=np.float32)
        return np.asarray(vector, dtype=np.float32).reshape(-1)

    async def embed(self, x: FloatArray, spans: list[tuple[int, int]]) -> FloatArray:
        """One vector per span of ``x`` (at :attr:`rate`): the mean of the span's pieces, L2-normalised for CLAP
        and speaker embeddings (tagger probabilities are kept as they are)."""
        size = max(1, int(self.clip_s * self.rate))
        floor = max(1, self.rate // 10)
        out: list[FloatArray] = []
        for a, b in spans:
            pieces = [x[s:e] for s, e in _pieces(a, b, size)]
            pieces = [np.pad(p, (0, floor - len(p))) if len(p) < floor else p for p in pieces]
            vectors = await asyncio.gather(*(self._one(np.ascontiguousarray(p, dtype=np.float32)) for p in pieces))
            mean = np.mean(np.stack(vectors), axis=0).astype(np.float32)
            out.append(mean if self.how == "tags" else _unit(mean))
        return np.stack(out)


# -- zero-shot ---------------------------------------------------------------------------------------------------
async def zero_shot(
    core: Bolo,
    samples: FloatArray,
    rate: int,
    model: str,
    labels: list[str],
    trace: Trace,
    prompt: str | None = None,
    window: float | None = None,
    hop: float | None = None,
) -> Classification:
    """Score ``samples`` against ``labels`` with the CLAP model ``model`` (resolved), whole and, with ``window``,
    window by window."""
    if len(labels) < 2:
        raise InvalidRequest("give at least two labels (a softmax over one label is always 1)", param="labels")
    duration = len(samples) / rate if rate else 0.0
    _check_window(window, hop, duration)
    async with core.manager.use(model) as entry:
        if not isinstance(entry.model, AudioTextModel):
            raise InvalidRequest(f"model '{model}' is not a zero-shot (CLAP) audio classifier", param="model")
        clap = entry.model
        enc = Encoder(entry)
        template = clap.prompt if prompt is None else prompt
        with trace.phase("embed_text"):
            texts = [prompt_text(template, label) for label in labels]
            text_vectors, _ = await entry.queue.submit("embed_text", texts)
        with trace.phase("embed_audio"):
            x = resample(np.asarray(samples, dtype=np.float32), rate, enc.rate) if rate != enc.rate else samples
            whole = await enc.embed(x, [(0, len(x))])
            spans = _spans(len(x), enc.rate, window, hop or window) if window is not None else []
            per = await enc.embed(x, spans) if spans else np.zeros((0, clap.dim), np.float32)
    t = np.asarray(text_vectors, dtype=np.float32)

    def scored(a: FloatArray) -> list[tuple[str, float, float | None]]:
        sims = t @ a
        probs = softmax(clap.logit_scale * sims)
        out: list[tuple[str, float, float | None]] = [
            (label, float(p), float(s)) for label, p, s in zip(labels, probs, sims)
        ]
        out.sort(key=lambda r: -r[1])
        return out

    result = Classification(model, duration, scored(whole[0]), prompt=template)
    result.windows = [Window(a / enc.rate, b / enc.rate, scored(v)) for (a, b), v in zip(spans, per)]
    return result


def _check_window(window: float | None, hop: float | None, duration: float) -> None:
    if window is not None and not 0.5 <= window <= 60.0:
        raise InvalidRequest("window must be between 0.5 and 60 seconds", param="window")
    if hop is not None and (window is None or not 0.1 <= hop <= window):
        raise InvalidRequest("hop must be between 0.1 s and the window", param="hop")
    if duration < 0.1:
        raise InvalidRequest("the recording is shorter than 0.1 s", param="file")


# -- trained heads -----------------------------------------------------------------------------------------------
@dataclass
class Head:
    """A classifier on embeddings: standardise, then logistic regression (``hidden`` empty) or one ReLU layer and
    a linear layer, then softmax."""

    labels: list[str]
    mean: FloatArray
    std: FloatArray
    weights: list[tuple[FloatArray, FloatArray]]

    def probabilities(self, x: FloatArray) -> FloatArray:
        h = (np.atleast_2d(x) - self.mean) / self.std
        for i, (w, b) in enumerate(self.weights):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        return softmax(h.astype(np.float32))

    def save(self, file: Path) -> None:
        arrays: dict[str, Any] = {"mean": self.mean, "std": self.std}
        for i, (w, b) in enumerate(self.weights):
            arrays[f"w{i}"], arrays[f"b{i}"] = w, b
        with file.open("wb") as fh:
            np.savez(fh, **arrays)

    @classmethod
    def load(cls, file: Path, labels: list[str]) -> Head:
        with np.load(file, allow_pickle=False) as data:
            weights = []
            i = 0
            while f"w{i}" in data:
                weights.append((data[f"w{i}"].astype(np.float32), data[f"b{i}"].astype(np.float32)))
                i += 1
            return cls(labels, data["mean"].astype(np.float32), data["std"].astype(np.float32), weights)


def fit_head(
    x: FloatArray, y: np.ndarray, labels: list[str], kind: str = "logistic", seed: int = 0, l2: float = 1e-3
) -> Head:
    """Train a head on embeddings ``x`` ``[n, d]`` with class indices ``y``: full-batch Adam on the cross-entropy
    plus an L2 penalty, from a fixed seed (the same data gives the same head)."""
    n, d = x.shape
    k = len(labels)
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0), 1e-4).astype(np.float32)
    z = ((x - mean) / std).astype(np.float64)
    onehot = np.eye(k)[y]
    rng = np.random.default_rng(seed)
    if kind == "mlp":
        hidden = 64
        params = [
            rng.normal(0, np.sqrt(2.0 / d), (d, hidden)), np.zeros(hidden),
            rng.normal(0, np.sqrt(1.0 / hidden), (hidden, k)), np.zeros(k),
        ]  # fmt: skip
        steps, lr = 400, 0.01
    else:
        params = [np.zeros((d, k)), np.zeros(k)]
        steps, lr = 300, 0.05
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for step in range(1, steps + 1):
        if kind == "mlp":
            pre = z @ params[0] + params[1]
            h = np.maximum(pre, 0.0)
            logits = h @ params[2] + params[3]
        else:
            logits = z @ params[0] + params[1]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        g = (p - onehot) / n
        if kind == "mlp":
            gh = (g @ params[2].T) * (pre > 0)
            grads = [z.T @ gh + l2 * params[0], gh.sum(0), h.T @ g + l2 * params[2], g.sum(0)]
        else:
            grads = [z.T @ g + l2 * params[0], g.sum(0)]
        for i, gr in enumerate(grads):
            m[i] = 0.9 * m[i] + 0.1 * gr
            v[i] = 0.999 * v[i] + 0.001 * gr * gr
            params[i] -= lr * (m[i] / (1 - 0.9**step)) / (np.sqrt(v[i] / (1 - 0.999**step)) + 1e-8)
    pairs = [(params[i].astype(np.float32), params[i + 1].astype(np.float32)) for i in range(0, len(params), 2)]
    return Head(labels, mean, std, pairs)


def cross_validate(x: FloatArray, y: np.ndarray, labels: list[str], kind: str) -> dict[str, Any] | None:
    """Stratified k-fold accuracy (k = 5, or fewer when a class has fewer examples); None when a class has only
    one example."""
    counts = np.bincount(y, minlength=len(labels))
    folds = min(5, int(np.min(counts)))
    if folds < 2:
        return None
    rng = np.random.default_rng(0)
    fold_of = np.zeros(len(y), dtype=int)
    for c in range(len(labels)):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % folds
    predicted = np.zeros(len(y), dtype=int)
    for f in range(folds):
        test = fold_of == f
        head = fit_head(x[~test], y[~test], labels, kind)
        predicted[test] = np.argmax(head.probabilities(x[test]), axis=1)
    correct = predicted == y
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y, predicted):
        confusion[t, p] += 1
    return {
        "accuracy": round(float(correct.mean()), 4),
        "folds": folds,
        "per_class": {label: round(float(correct[y == c].mean()), 4) for c, label in enumerate(labels)},
        "confusion": confusion.tolist(),
    }


def check_classifier_id(name: str) -> str:
    cid = name.strip().lower()
    if not _ID_RE.match(cid):
        raise InvalidRequest(
            f"invalid classifier name {name!r}: lowercase letters, digits, '.', '-' and '_' (up to 64)", param="name"
        )
    return cid


def _meta_file(core: Bolo, cid: str) -> Path:
    return core.store.model_dir(cid) / "classifier.json"


def is_classifier(spec: ModelSpec | None) -> bool:
    return spec is not None and spec.kind == HEAD_KIND


def classifier_ids(core: Bolo) -> list[str]:
    return [s.id for s in core.catalog.all() if s.kind == HEAD_KIND]


def describe(core: Bolo, cid: str) -> dict[str, Any]:
    spec = core.catalog.resolve(cid)
    if not is_classifier(spec):
        raise ClassifierNotFound(f"no trained classifier '{cid}' (bolo classifier list)", param="classifier")
    assert spec is not None
    try:
        meta: dict[str, Any] = json.loads(_meta_file(core, spec.id).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClassifierNotFound(
            f"the files of classifier '{spec.id}' are missing or damaged", param="classifier"
        ) from exc
    return {"object": "audio.classifier", **meta}


def list_classifiers(core: Bolo) -> list[dict[str, Any]]:
    out = []
    for cid in classifier_ids(core):
        try:
            out.append(describe(core, cid))
        except ClassifierNotFound:
            continue
    return out


_heads: dict[str, tuple[float, Head]] = {}


def load_head(core: Bolo, cid: str) -> tuple[Head, dict[str, Any]]:
    meta = describe(core, cid)
    file = core.store.model_dir(cid) / "head.npz"
    stamp = file.stat().st_mtime if file.is_file() else -1.0
    cached = _heads.get(cid)
    if cached is None or cached[0] != stamp:
        if stamp < 0:
            raise ClassifierNotFound(f"the weights of classifier '{cid}' are missing", param="classifier")
        _heads[cid] = (stamp, Head.load(file, [str(x) for x in meta["labels"]]))
    return _heads[cid][1], meta


async def classify_trained(
    core: Bolo,
    samples: FloatArray,
    rate: int,
    cid: str,
    trace: Trace,
    window: float | None = None,
    hop: float | None = None,
) -> Classification:
    """Classify ``samples`` with the trained classifier ``cid``: its encoder's embeddings, then its head."""
    head, meta = load_head(core, cid)
    duration = len(samples) / rate if rate else 0.0
    _check_window(window, hop, duration)
    encoder = str(meta["encoder"])
    async with core.manager.use(encoder) as entry:
        enc = Encoder(entry)
        with trace.phase("embed_audio"):
            x = resample(np.asarray(samples, dtype=np.float32), rate, enc.rate) if rate != enc.rate else samples
            whole = await enc.embed(x, [(0, len(x))])
            spans = _spans(len(x), enc.rate, window, hop or window) if window is not None else []
            per = await enc.embed(x, spans) if spans else None
    dim = int(meta.get("dim") or head.mean.shape[0])
    if whole.shape[1] != dim:
        raise InvalidRequest(
            f"classifier '{cid}' was trained on {dim}-number embeddings and '{encoder}' now gives {whole.shape[1]}: "
            "train it again",
            param="classifier",
        )

    def scored(v: FloatArray) -> list[tuple[str, float, float | None]]:
        probs = head.probabilities(v)[0]
        out: list[tuple[str, float, float | None]] = [(label, float(p), None) for label, p in zip(head.labels, probs)]
        out.sort(key=lambda r: -r[1])
        return out

    result = Classification(encoder, duration, scored(whole[0]), classifier=cid)
    if per is not None:
        result.windows = [Window(a / enc.rate, b / enc.rate, scored(v)) for (a, b), v in zip(spans, per)]
    return result


async def train(
    core: Bolo,
    name: str,
    examples: list[tuple[str, FloatArray, int]],
    encoder: str | None = None,
    kind: str = "logistic",
    replace: bool = False,
    description: str | None = None,
) -> dict[str, Any]:
    """Train and save the classifier ``name`` from ``examples`` (label, samples, rate); returns its description
    with the cross-validated accuracy and warnings (too few examples, unbalanced classes, low accuracy)."""
    from bolo.catalog import CatalogError, ModelSpec
    from bolo.store import Source
    from bolo.voices import now_iso

    cid = check_classifier_id(name)
    if kind not in ("logistic", "mlp"):
        raise InvalidRequest("head must be 'logistic' or 'mlp'", param="head")
    existing = core.catalog.resolve(cid)
    if existing is not None and not is_classifier(existing):
        raise ClassifierExists(f"'{cid}' is already the id of a model; choose another name", param="name")
    if existing is not None and not replace:
        raise ClassifierExists(f"a classifier called '{cid}' exists (replace it with replace=true)", param="name")
    if not examples:
        raise InvalidRequest("no examples: give audio clips, each with its class label", param="file")
    if len(examples) > MAX_EXAMPLES:
        raise InvalidRequest(f"at most {MAX_EXAMPLES} examples per classifier", param="file")
    labels = sorted({label for label, _, _ in examples})
    if len(labels) < 2:
        raise InvalidRequest("examples of at least two classes are needed", param="label")
    spec = core.resolve(encoder or None, TASK) if not encoder else core.manager.resolve(encoder)
    if spec.task not in ENCODER_TASKS or is_classifier(spec):
        raise InvalidRequest(
            f"'{spec.id}' cannot be an encoder: use a CLAP ({TASK}), speaker-embedding or audio-tagging model",
            param="encoder",
        )
    started = time.monotonic()
    for label, samples, rate in examples:
        if rate <= 0 or len(samples) < rate // 10:
            raise InvalidRequest(f"an example of '{label}' is shorter than 0.1 s", param="file")
    vectors: list[FloatArray] = []
    async with core.manager.use(spec.id) as entry:
        enc = Encoder(entry)

        async def one(samples: FloatArray, rate: int) -> FloatArray:
            x = resample(np.asarray(samples, dtype=np.float32), rate, enc.rate) if rate != enc.rate else samples
            first: FloatArray = (await enc.embed(x, [(0, len(x))]))[0]
            return first

        # a few examples at a time, so that the encoder's queue can batch them
        for i in range(0, len(examples), EMBED_AT_ONCE):
            group = examples[i : i + EMBED_AT_ONCE]
            vectors.extend(await asyncio.gather(*(one(samples, rate) for _, samples, rate in group)))
    embed_s = time.monotonic() - started
    x_all = np.stack(vectors).astype(np.float32)
    y = np.array([labels.index(label) for label, _, _ in examples])
    t0 = time.monotonic()
    cv = await asyncio.to_thread(cross_validate, x_all, y, labels, kind)
    head = await asyncio.to_thread(fit_head, x_all, y, labels, kind)
    train_s = time.monotonic() - t0
    train_acc = float((np.argmax(head.probabilities(x_all), axis=1) == y).mean())
    counts = {label: int((y == i).sum()) for i, label in enumerate(labels)}
    warnings = []
    few = [label for label, c in counts.items() if c < FEW_EXAMPLES]
    if few:
        warnings.append(
            f"too little data: {', '.join(few)} {'has' if len(few) == 1 else 'have'} fewer than {FEW_EXAMPLES} "
            "examples; a few dozen per class make a classifier you can trust"
        )
    if max(counts.values()) > 3 * min(counts.values()):
        warnings.append("unbalanced classes: one class has more than three times the examples of another")
    if cv is None:
        warnings.append("no cross-validation: a class has a single example, so the accuracy is unknown")
    elif cv["accuracy"] < 0.7:
        warnings.append(
            f"cross-validated accuracy is only {cv['accuracy']:.0%}: more or cleaner examples, or another encoder, "
            "may help"
        )
    mdir = core.store.model_dir(cid)
    await core.manager.unload(cid, reason="retrained")
    if existing is not None:
        await asyncio.to_thread(core.store.remove, cid)
        core.catalog.remove_custom(cid)
    mdir.mkdir(parents=True, exist_ok=True)
    head.save(mdir / "head.npz")
    meta: dict[str, Any] = {
        "id": cid,
        "labels": labels,
        "encoder": spec.id,
        "encoder_task": spec.task,
        "head": kind,
        "dim": int(x_all.shape[1]),
        "examples": counts,
        "cross_validation": cv,
        "train_accuracy": round(train_acc, 4),
        "warnings": warnings,
        "embed_s": round(embed_s, 2),
        "train_s": round(train_s, 3),
        "created_at": now_iso(),
    }
    if description:
        meta["description"] = description[:500]
    (mdir / "classifier.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    registered = core.store.register_local(cid, mdir, {"classifier": True})
    accuracy = f"; cross-validated accuracy {cv['accuracy']:.0%}" if cv else ""
    entry_spec = ModelSpec(
        id=cid,
        task=TASK,
        engine="builtin",
        kind=HEAD_KIND,
        family="Trained classifier",
        languages=[],
        tier="realtime",
        size_bytes=int(registered["size_bytes"]),
        license=spec.license,
        license_url=spec.license_url,
        commercial_use=spec.commercial_use,
        features=["trained"],
        description=(description or "")[:500]
        or f"Trained here from {len(examples)} examples of {', '.join(labels)} on {spec.id} embeddings{accuracy}.",
        source_url="",
        validated="untested",
        source=Source(local=str(mdir)),
        options={"encoder": spec.id, "labels": labels, "head": kind},
    )
    try:
        core.catalog.add_custom(entry_spec)
    except CatalogError as exc:
        raise ClassifierExists(str(exc), param="name") from exc
    _heads.pop(cid, None)
    core.events.publish("model", {"id": cid, "event": "installed"})
    return {"object": "audio.classifier", **meta}


async def delete(core: Bolo, cid: str) -> None:
    spec = core.catalog.resolve(cid)
    if not is_classifier(spec):
        raise ClassifierNotFound(f"no trained classifier '{cid}'", param="classifier")
    assert spec is not None
    _heads.pop(spec.id, None)
    await core.remove(spec.id)


# -- response ----------------------------------------------------------------------------------------------------
def body(result: Classification, top_k: int | None = None) -> dict[str, Any]:
    def scores(rows: list[tuple[str, float, float | None]]) -> list[dict[str, Any]]:
        out = []
        for label, score, sim in rows[: top_k or len(rows)]:
            item: dict[str, Any] = {"label": label, "score": round(score, 4)}
            if sim is not None:
                item["similarity"] = round(sim, 4)
            out.append(item)
        return out

    def best(rows: list[tuple[str, float, float | None]]) -> dict[str, Any]:
        label, score, _ = rows[0] if rows else (None, 0.0, None)
        return {"label": label, "score": round(score, 4), "scores": scores(rows)}

    out: dict[str, Any] = {
        "object": "audio.classification",
        "model": result.model,
        "classifier": result.classifier,
        "duration": round(result.duration, 3),
        **best(result.scores),
    }
    if result.prompt is not None:
        out["prompt"] = result.prompt
    if result.windows:
        out["segments"] = [
            {"start": round(w.start, 3), "end": round(w.end, 3), **best(w.scores)} for w in result.windows
        ]
    return out


async def classify_request(
    core: Bolo,
    samples: FloatArray,
    rate: int,
    trace: Trace,
    *,
    model: str | None = None,
    classifier: str | None = None,
    labels: list[str] | None = None,
    prompt: str | None = None,
    window: float | None = None,
    hop: float | None = None,
) -> Classification:
    """What ``POST /v1/audio/classify`` and ``bolo classify`` do: a trained classifier (``classifier``, or a
    ``model`` that is one), else zero-shot against ``labels`` with a CLAP model (``model`` or the default)."""
    if classifier:
        spec = core.catalog.resolve(classifier)
        if not is_classifier(spec):
            raise ClassifierNotFound(f"no trained classifier '{classifier}' (bolo classifier list)", param="classifier")
        assert spec is not None
        return await classify_trained(core, samples, rate, spec.id, trace, window, hop)
    spec = core.resolve(model or None, TASK)
    if is_classifier(spec):
        return await classify_trained(core, samples, rate, spec.id, trace, window, hop)
    if not labels:
        raise InvalidRequest("give labels (the classes, in plain words) or a trained classifier", param="labels")
    return await zero_shot(core, samples, rate, spec.id, labels, trace, prompt, window, hop)
