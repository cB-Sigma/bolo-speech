#!/usr/bin/env python3
"""Check the demo's MOSS-TTS-Nano driver (ui/src/lib/browser-clone/moss*.ts) against the model's own
reference runtime: the official ONNX runtime from github.com/OpenMOSS/MOSS-TTS-Nano (Apache-2.0,
ort_cpu_runtime.py and tts_robust_normalizer_single_script.py), on the same ONNX files.

    git clone https://github.com/OpenMOSS/MOSS-TTS-Nano /tmp/moss-src && git -C /tmp/moss-src checkout 8b7bcc93
    python -m venv /tmp/v && /tmp/v/bin/pip install numpy onnxruntime sentencepiece

The model directory holds the two Hub repositories at the revisions in models.ts, side by side, as the
reference runtime expects them (``MOSS-TTS-Nano-100M-ONNX/`` and ``MOSS-Audio-Tokenizer-Nano-ONNX/``).

1. ``vectors``: token ids from SentencePiece for a set of sentences in the model's languages, written as
   ``moss-tokenizer-vectors.json`` next to ``tokenizer.model``. ``BOLO_MOSS_TOKENIZER=<tokenizer.model>
   npx vitest run spm-bpe`` compares the page's tokenizer with them.
2. ``refs --out DIR``: the bundled example voices (ui/public/clone/*.mp3) as 24 kHz float32 files (ffmpeg).
3. ``BOLO_MOSS_DIR=<model dir> BOLO_MOSS_OUT=DIR npx vitest run moss.parity`` runs the page's driver with
   ONNX Runtime Web (WebAssembly, in Node) and writes, per voice, text and seed, the reference codes, prompt
   rows, the uniform numbers it drew, the frames and the audio.
4. ``compare --out DIR``: runs the reference runtime on the same 48 kHz input with the same uniform numbers
   and prints whether the codes, rows and frames are equal and how close the audio is.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent.parent / "public" / "clone"
TTS = "MOSS-TTS-Nano-100M-ONNX"
CODEC = "MOSS-Audio-Tokenizer-Nano-ONNX"

SENTENCES = [
    "Hello, how are you today?",
    "  leading and   trailing  spaces  ",
    "tabs\tand\nnew lines\r\nhere",
    "Ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３ and ﬁ ligature ½ ①",
    "Emoji 😀👍🏽 and 𝔘𝔫𝔦𝔠𝔬𝔡𝔢",
    "zero\u200bwidth\ufeffchars",
    "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。",
    "Vielen Dank für Ihren Anruf. Ihr Termin am Dienstag um halb elf ist bestätigt.",
    "Gracias por llamar. Su cita del martes a las diez y media está confirmada.",
    "Merci de votre appel. Votre rendez-vous de mardi à dix heures et demie est confirmé.",
    "こんにちは、今日はいい天気ですね。カタカナもテスト。",
    "שלום, מה שלומך היום?",
    "안녕하세요, 오늘 날씨가 좋네요.",
    "Привет, как дела? Всё хорошо!",
    "سلام، حال شما چطور است؟",
    "مرحبا، كيف حالك اليوم؟",
    "Dzień dobry, jak się masz? Źdźbło żółć.",
    "Dobrý den, jak se máte? Příliš žluťoučký kůň.",
    "Tack för samtalet. Åsa äter räksmörgås.",
    "Köszönöm a hívást. Árvíztűrő tükörfúrógép.",
    "Ευχαριστώ για την κλήση. Καλημέρα σας!",
    "Aradığınız için teşekkürler. Çok güzel.",
    "Numbers 123,456.78 and 10:30 am, 2024-05-01, $5.99, 50% off!",
    "<|im_start|>user<|im_end|> literal special tokens",
    "Mixed中文English混排text。",
    "\u3000ideographic\u3000space",
    "Ⅻ ㎏ ㍿ ﹖ ﹗ ｶﾀｶﾅ",
    "e\u0301 combining vs é precomposed",
]


def reference(src: Path):
    sys.path.insert(0, str(src))
    import ort_cpu_runtime  # noqa: PLC0415
    import tts_robust_normalizer_single_script  # noqa: PLC0415

    return ort_cpu_runtime, tts_robust_normalizer_single_script


def vectors(model_dir: Path) -> None:
    import sentencepiece as spm  # noqa: PLC0415

    path = model_dir / TTS / "tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=str(path))
    out = [{"text": t, "ids": [int(i) for i in sp.encode(t)]} for t in SENTENCES]
    (path.parent / "moss-tokenizer-vectors.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"{len(out)} sentences -> {path.parent / 'moss-tokenizer-vectors.json'}")


def refs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for mp3 in sorted(EXAMPLES.glob("*.mp3")):
        target = out / f"{mp3.stem}.f32"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", "24000", "-f", "f32le", str(target)],
            check=True,
        )
        print(target)


class Uniforms:
    """The uniform numbers the page drew, handed out in the same order (the reference calls rng.random())."""

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.i = 0

    def random(self) -> float:
        # Past the page's last frame (the runs went apart): any number will do, the frames already differ.
        v = self.values[self.i] if self.i < len(self.values) else 0.5
        self.i += 1
        return v


def decimate(x: np.ndarray) -> np.ndarray:
    n = np.arange(-63, 64)
    h = 0.46 * np.sinc(0.46 * n) * np.blackman(127)
    return np.convolve(x, h / h.sum(), mode="same")[::2]


def compare(model_dir: Path, src: Path, out: Path, threads: int) -> None:
    ort_cpu_runtime, _ = reference(src)
    import sentencepiece as spm  # noqa: PLC0415

    rt = ort_cpu_runtime.OrtCpuRuntime(str(model_dir / TTS), thread_count=threads)
    sp = spm.SentencePieceProcessor(model_file=str(model_dir / TTS / "tokenizer.model"))
    for dump in sorted(out.glob("*.js.json")):
        tag = dump.name[: -len(".js.json")]
        j = json.loads(dump.read_text())
        wav = np.fromfile(out / f"{j['voice']}.48.f32", np.float32).reshape(2, -1)[None]
        enc = rt.sessions["codec_encode"].run(
            None, {"waveform": wav, "input_lengths": np.asarray([wav.shape[-1]], np.int32)}
        )
        codes = enc[0][0, : int(enc[1][0])].tolist()
        rt.rng = Uniforms(j["uniforms"])
        frames: list[list[int]] = []
        audio: list[np.ndarray] = []
        rows_equal = True
        for i, chunk in enumerate(j["chunks"]):
            ids = [int(x) for x in sp.encode(chunk["text"])]
            if ids != chunk["ids"]:
                raise SystemExit(f"{tag}: token ids differ for {chunk['text']!r}")
            request = rt.build_voice_clone_request_rows(j["codes"], ids)
            rows_equal &= np.asarray(request["inputIds"]).reshape(-1).tolist() == j["rows"][i]
            got = rt.generate_audio_frames(request)
            frames += got
            rt.codec_streaming_session.reset()
            for f in got:
                decoded = rt.codec_streaming_session.run_frames([f])
                if decoded:
                    audio.append(decoded[0][0, :, : decoded[1]].mean(0))
            if chunk["pause"]:
                audio.append(np.zeros(round(48000 * chunk["pause"]), np.float32))
        py = decimate(np.concatenate(audio)) if audio else np.zeros(0, np.float32)
        js = np.fromfile(out / f"{tag}.js.f32", np.float32)
        n = min(len(py), len(js))
        lag = max(range(-64, 65), key=lambda k: float(np.dot(py[max(0, k) : n + min(0, k)], js[max(0, -k) : n - max(0, k)])))
        a, b = py[max(0, lag) : n + min(0, lag)], js[max(0, -lag) : n - max(0, lag)]
        first_diff = next((i for i, (x, y) in enumerate(zip(frames, j["frames"])) if x != y), None)
        print(
            f"{tag}: codes {'equal' if codes == j['codes'] else 'DIFFER'}, rows {'equal' if rows_equal else 'DIFFER'}, "
            f"frames {len(frames)} (page {len(j['frames'])}), first differing frame {first_diff}, "
            f"audio correlation {np.corrcoef(a, b)[0, 1]:.5f}, relative RMS difference "
            f"{np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a**2)) + 1e-9):.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["vectors", "refs", "compare"])
    ap.add_argument("--model-dir", type=Path, help="the two Hub repositories side by side")
    ap.add_argument("--src", type=Path, help="a checkout of OpenMOSS/MOSS-TTS-Nano")
    ap.add_argument("--out", type=Path, help="where the page's driver writes its runs")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.step == "vectors":
        vectors(a.model_dir)
    elif a.step == "refs":
        refs(a.out)
    else:
        compare(a.model_dir, a.src, a.out, a.threads)


if __name__ == "__main__":
    main()
