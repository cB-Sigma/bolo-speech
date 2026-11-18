"""Write what the server's punctuation model answers for a few hundred transcripts, so the page's TypeScript port
(src/lib/browser-punct) can be held to the same results, word for word (src/lib/browser-punct/punct.test.ts).

    python ui/scripts/punctuation/export_cases.py MODEL_DIR      # from the repository root

MODEL_DIR is the unpacked punct-en-cnn-bilstm (sherpa-onnx-online-punct-en-2024-08-06: model.int8.onnx and
bpe.vocab). Needs the server's sources, sherpa-onnx and ONNX Runtime.

For every text the fixture holds what the server returns (bolo.engines.sherpa.SherpaPunctuation, after the
lower-casing of all-capital text in bolo.punctuation.apply), and the steps in between as this script reproduces
them in Python: the token ids and the label positions of each chunk, as sherpa-onnx's simple-sentencepiece and
OnlinePunctuationCNNBiLSTMImpl build them, and the case and punctuation the model picks for each word. The script
stops if its own steps do not give exactly the server's answer, so the ids in the fixture are sherpa-onnx's.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "src"))

from bolo.engines.base import LoadOptions  # noqa: E402
from bolo.engines.sherpa import SherpaPunctuation  # noqa: E402

OUT = ROOT / "ui" / "src" / "lib" / "browser-punct" / "fixtures" / "server-punct-cases.json"
VOCAB_OUT = ROOT / "ui" / "src" / "lib" / "browser-punct" / "fixtures" / "bpe.vocab.gz"
MAX_SEQ = 200  # kMaxSeqLen
C_SPACE = " \t\n\v\f\r"  # what `std::istream >> std::string` splits on (the "C" locale)


class Ssentencepiece:
    """simple-sentencepiece 0.7 (ssentencepiece.cc): the best-scoring cut of the UTF-8 bytes, float32 sums."""

    def __init__(self, vocab: str) -> None:
        self.tokens: list[bytes] = []
        self.scores: list[np.float32] = []
        for line in vocab.splitlines():
            token, score = line.split()[:2]
            self.tokens.append(token.encode())
            self.scores.append(np.float32(score))
        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self.longest = max(len(t) for t in self.tokens)
        self.unk = self.tokens.index(b"<unk>")

    def encode(self, word: str) -> list[int]:
        s = b"".join(b"\xe2\x96\x81" + w.encode() for w in c_split(word))
        n = len(s)
        route: list[tuple[np.float32, int, int]] = [(np.float32(0), 0, 0)] * (n + 1)
        for i in range(n - 1, -1, -1):
            best, end, idx = np.float32(-np.inf), -1, 0
            for j in range(i + 1, min(n, i + self.longest) + 1):
                tid = self.ids.get(s[i:j])
                if tid is None:
                    continue
                score = np.float32(self.scores[tid] + route[j][0])
                if score > best or (score == best and end >= j):
                    best, end, idx = score, j, tid
            route[i] = (np.float32(0) if best == -np.inf else best, end, idx)
        out: list[int] = []
        i = 0
        while i < n:
            if route[i][1] == -1:
                out.append(self.unk)
                i += 1
            else:
                out.append(route[i][2])
                i = route[i][1]
        return out


def too_long(text: str) -> bool:
    """A word of more than 190 bytes: sherpa-onnx 2.0.1 crashes the process when one word needs more than a chunk
    of 200 tokens, so such texts are left out of the fixture (the page returns them as they were)."""
    return any(len(w.encode()) > 190 for w in c_split(text))


def c_split(text: str) -> list[str]:
    return [w for w in re.split(f"[{re.escape(C_SPACE)}]+", text) if w]


def encode_sentences(sp: Ssentencepiece, text: str) -> list[dict[str, list[int]]]:
    """OnlinePunctuationCNNBiLSTMImpl::EncodeSentences: chunks of at most 200 tokens, <s> 1 and </s> 2."""
    chunks: list[dict[str, list[int]]] = []
    tokens, valids = [1], [1]
    for word in c_split(text):
        pieces = sp.encode(word)
        if len(tokens) + len(pieces) > MAX_SEQ - 1:
            chunks.append({"ids": tokens + [2], "valid": valids + [1]})
            tokens, valids = [1], [1]
        tokens += pieces
        valids += [1] + [0] * (len(pieces) - 1)
    if tokens:
        chunks.append({"ids": tokens + [2], "valid": valids + [1]})
    return chunks


def run_model(session, chunks: list[dict[str, list[int]]]) -> tuple[list[int], list[int]]:
    n = len(chunks)
    ids = np.zeros((n, MAX_SEQ), np.int32)
    valid = np.zeros((n, MAX_SEQ), np.int32)
    lens = np.zeros((n,), np.int32)
    for k, c in enumerate(chunks):
        ids[k, : len(c["ids"])] = c["ids"]
        valid[k, : len(c["valid"])] = c["valid"]
        lens[k] = sum(c["valid"])
    case, punct = session.run(None, {"token_ids": ids, "valid_ids": valid, "label_lens": lens})[:2]
    return [int(x) for x in case.argmax(-1)], [int(x) for x in punct.argmax(-1)]


def decode(text: str, case: list[int], punct: list[int]) -> str:
    """OnlinePunctuationCNNBiLSTMImpl::DecodeSentences: case per word in the "C" locale, then the mark."""
    out = []
    for i, word in enumerate(c_split(text)):
        c = case[i] if i < len(case) else 0
        if c == 1:
            word = word.upper() if word.isascii() else "".join(ch.upper() if ch.isascii() else ch for ch in word)
        elif c == 2 and word[:1].isascii():
            word = word[:1].upper() + word[1:]
        out.append(word + {1: ",", 2: ".", 3: "?"}.get(punct[i] if i < len(punct) else 0, ""))
    return " ".join(out)


def corpus() -> list[str]:
    """Sentences of the project's own documentation with the punctuation and capitals taken away, and edge cases."""
    texts: list[str] = []
    seen: set[str] = set()
    for md in sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]:
        body = re.sub(r"```.*?```", " ", md.read_text(encoding="utf-8"), flags=re.S)
        for line in body.splitlines():
            if not line or line.lstrip().startswith(("#", "|", "<", ">", "-", "*", "[", "!")):
                continue
            line = re.sub(r"`[^`]*`|\[([^\]]*)\]\([^)]*\)|https?://\S+", r"\1", line)
            for sent in re.split(r"(?<=[.?!])\s+", line):
                words = re.findall(r"[A-Za-z][A-Za-z']*|\d+", sent)
                if not 4 <= len(words) <= 40:
                    continue
                t = " ".join(words).lower()
                if t not in seen:
                    seen.add(t)
                    texts.append(t)
    picked = texts[:: max(1, len(texts) // 260)][:260]
    long = " ".join(picked[:40])  # several chunks of 200 tokens
    edge = [
        "",
        "   ",
        "hello",
        "HELLO HOW ARE YOU DOING TODAY",
        "Hello how are you doing today",
        "i think mcdonald's is open on sundays isn't it",
        "  leading and trailing spaces  ",
        "tabs\tand\nnew lines\r\nin between  words",
        "non breaking space stays inside a word",
        "what time is it in new york right now",
        "the price went up by 25 percent in 2024 and 3 more next year",
        "naïve café owners in zürich serve crème brûlée",
        "ångström units and ß and émile",
        "emoji 😀 in the middle of a sentence",
        "मैं ठीक हूँ thank you very much",
        "<unk> <s> </s> tokens written out",
        "supercalifragilisticexpialidocious antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis",
        "a" * 250,
        "the longest word it takes " + "b" * 190 + " is fine",
        "x\u00a0" * 100,
        "x " * 230,
        "where are you going i am going home are you coming with me",
        "　ideographic space at the start",
        "U.S.A. and e.g. stay as they are",
        "it's 5 o'clock somewhere isn't it",
        long,
        long.upper(),
    ]
    return picked + edge


def main() -> None:
    import onnxruntime as ort

    model_dir = Path(sys.argv[1])
    vocab = (model_dir / "bpe.vocab").read_text(encoding="utf-8")
    sp = Ssentencepiece(vocab)
    server = SherpaPunctuation("punct_cnn_bilstm", model_dir, {}, LoadOptions(threads=1))
    session = ort.InferenceSession(str(model_dir / "model.int8.onnx"), providers=["CPUExecutionProvider"])
    cases = []
    for raw in corpus():
        text = raw.lower() if raw.isupper() else raw  # bolo.punctuation.apply, for models that write capitals
        if too_long(text):
            continue
        expected = server.punctuate(text)
        case: dict[str, object] = {"text": raw, "expected": expected}
        if text.strip():
            chunks = encode_sentences(sp, text)
            cpred, ppred = run_model(session, chunks)
            mine = decode(text, cpred, ppred).strip()
            if mine != expected:
                sys.exit(f"the Python steps disagree with the server on {raw!r}:\n  {mine!r}\n  {expected!r}")
            compact = [{"ids": c["ids"], "valid": "".join(map(str, c["valid"]))} for c in chunks]
            case.update(chunks=compact, case="".join(map(str, cpred)), punct="".join(map(str, ppred)))
        cases.append(case)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": "punct-en-cnn-bilstm", "cases": cases}, ensure_ascii=False) + "\n")
    import gzip

    VOCAB_OUT.write_bytes(gzip.compress(vocab.encode(), 9, mtime=0))
    sys.stdout.write(f"{len(cases)} cases -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)\n")


if __name__ == "__main__":
    main()
