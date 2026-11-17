"""Write the server's answers for the text checks the page repeats in TypeScript, so the unit tests can hold the
page to the same results (src/lib/langdetect.test.ts, src/lib/repeat-guard.test.ts).

    python ui/scripts/parity/export_cases.py      # from the repository root; needs only the server's sources

Language detection: every text of the server's own tests (server/tests/test_langdetect.py, read with `ast`, so
new cases there are picked up) and a corpus of harder ones, each with what bolo.text.langdetect.detect answers.
The repetition guard: the server tests' texts and sequences of live finals, with what trim_repeats and RepeatGuard
answer. The feature front ends: server-features.json (see export_features). Rerun after changing either module on the server, and commit the JSON.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "src"))

from bolo.engines.base import Segment  # noqa: E402
from bolo.stt_options import RepeatGuard, trim_repeats  # noqa: E402
from bolo.text.langdetect import detect  # noqa: E402

OUT = ROOT / "ui" / "src" / "lib" / "fixtures" / "server-text-cases.json"
FEATURES = ROOT / "ui" / "src" / "lib" / "fixtures" / "server-features.json"


def export_features() -> None:
    """Whisper's log-mel (Smart Turn) and SpeechBrain's 60-band filterbank (VoxLingua107) of a test signal, as the
    server computes them (src/lib/browser-audio/dsp.test.ts). Needs NumPy and ONNX Runtime (bolo.engines.lid_onnx)."""
    import numpy as np

    from bolo.audio.features import whisper_log_mel
    from bolo.engines.lid_onnx import fbank60

    i = np.arange(16000)
    x = (np.sin(i / 7) * 0.3 + 0.1 * np.sin(i / 3.1) + 0.05 * np.sin(i / 0.9)).astype(np.float32)
    w = whisper_log_mel(x, normalize=True)
    f = fbank60(x)[0]
    out = {
        "whisper": {
            "shape": list(w.shape),
            "col0": [round(float(v), 5) for v in w[:, 0]],
            "col50": [round(float(v), 5) for v in w[:, 50]],
            "sum": float(w.sum()),
        },
        "fbank60": {
            "shape": list(f.shape),
            "row0": [round(float(v), 4) for v in f[0]],
            "row30": [round(float(v), 4) for v in f[30]],
            "sum": float(f.sum()),
        },
    }
    FEATURES.write_text(json.dumps(out, indent=0) + "\n", encoding="utf-8")


def server_test_texts() -> list[dict[str, object]]:
    """The strings of server/tests/test_langdetect.py's parametrised tests and the ones passed to detect()."""
    tree = ast.parse((ROOT / "server" / "tests" / "test_langdetect.py").read_text(encoding="utf-8"))
    cases: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "parametrize":
            names = ast.literal_eval(node.args[0])
            for row in ast.literal_eval(node.args[1]):
                if isinstance(names, tuple) and len(names) == 3:
                    cases.append({"text": row[0], "expect": {"language": row[1], "script": row[2]}})
                else:
                    cases.append({"text": row, "expect": {"language": None}})
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "detect" and node.args:
            try:
                cases.append({"text": ast.literal_eval(node.args[0])})
            except ValueError:
                pass
    return cases


CORPUS = [
    "The weather is lovely this morning, and we will go out for a walk.",
    "I think it is what it is.",
    "Hej, jeg hedder Anna og jeg kan godt lide at læse bøger.",
    "Hei, jeg heter Anna og jeg liker å lese bøker, ikkje sant.",
    "Jag heter Anna och jag tycker om att läsa böcker.",
    "Minä olen kotona ja hän on töissä, mutta me tulemme myöhemmin.",
    "Dobrý den, jak se máte? Já jsem v pořádku a není to tak špatné.",
    "Bună ziua, ce mai faci? Eu sunt bine și mă bucur să te văd.",
    "Jó napot, hogy van? Nem tudom, hogy ez még kell-e.",
    "Selamat pagi, saya akan pergi ke pasar dengan ibu saya.",
    "Xin chào, tôi là người Việt Nam và tôi đã đến đây.",
    "Bon dia, com estàs? Jo estic molt bé, però és tard.",
    "Habari yako? Mimi ni mzima sana na wewe je?",
    "Ciao, sono qui per il lavoro e non per la vacanza.",
    "Добрий день! Як справи? Все добре, дякую.",
    "Добър ден, как сте? Аз съм добре и всички са тук.",
    "Здраво, како си? Ја сам добро, хвала.",
    "سلام، حال شما چطور است؟ من خوبم.",
    "मला माहित नाही की तो कुठे आहे.",
    "तिमी कहाँ जाँदै छौ? म घर जाँदै छु।",
    "আমি তোমাকে ভালোবাসি।",
    "ਮੈਂ ਘਰ ਜਾ ਰਿਹਾ ਹਾਂ।",
    "ଆପଣ କେମିତି ଅଛନ୍ତି?",
    "ᱥᱟᱱᱛᱟᱲᱤ",
    "東京は日本の首都です",
    "我爱北京天安门",
    "ありがとう",
    "Ελληνικά είναι ωραία",
    "Բարեւ Ձեզ",
    "გამარჯობა",
    "ສະບາຍດີ",
    "ជំរាបសួរ",
    "မင်္ဂလာပါ",
    "བཀྲ་ཤིས་བདེ་ལེགས།",
    "ሰላም እንዴት ነህ",
    "<speak><prosody rate='slow'>Bonjour, je suis très content de vous voir ici.</prosody></speak>",
    "[Kokoro](/kˈOkəɹO/) is a voice, and it is good at this.",
    "Mixed: नमस्ते and hello, this is a test of the system.",
    "Hello नमस्ते",
    "Straße, Größe, Fuß — ß ist deutsch.",
    "¿Qué?",
    "Ça va très bien, merci beaucoup.",
    "Zażółć gęślą jaźń.",
    "Tamam, şimdi geliyorum ve ılık bir çay içeceğiz.",
    "da da da da",
    "It’s a dog’s life, isn’t it? We don’t know.",
    "123 456 789",
    "!!! ??? ...",
    "ab cd ef",
    "ΟΔΟΣ ΣΟΦΙΑΣ",
    "Ⅻ century and ² of the things in the world",
    "क्या हाल है? सब ठीक है।",
    "நான் நன்றாக இருக்கிறேன்",
    "ఎలా ఉన్నారు",
    "ಹೇಗಿದ್ದೀರಿ",
    "സുഖമാണോ",
    "કેમ છો મિત્રો",
    "ꯃꯤꯇꯩꯂꯣꯟ",
    "Hallo, wie geht es dir? Ich bin müde, aber es ist schön.",
    "Hoje é um bom dia para ir à praia, não é?",
    "Een twee drie, dat is het, en ik ga naar huis.",
]


def detection(text: str) -> dict[str, object]:
    d = detect(text)
    return {
        "language": d.language,
        "script": d.script,
        "confidence": round(d.confidence, 6),
        "alternatives": list(d.alternatives),
        "reason": d.reason,
    }


TRIM = [
    ["So we went home and " + "the " * 40 + "end.", 4],
    ["thank you for watching " * 12, 4],
    ["No, no, no, no, no, no, I said.", 4],
    ["谢谢大家" * 10, 4],
    ["very very good", 4],
    ["no no no no", 4],
    ["it is what it is", 4],
    ["ha ha ha ha", 4],
    ["the " * 40, 0],
    ["the " * 40, 2],
    ["hello " + "you know " * 10 + "right", 4],
    ["one two three one two three one two three one two three one two three four", 4],
    ["ठीक है ठीक है ठीक है ठीक है ठीक है ठीक है", 4],
    ["ขอบคุณครับขอบคุณครับขอบคุณครับขอบคุณครับขอบคุณครับขอบคุณครับ", 4],
    ["The The the THE the, the. the! done", 4],
    ["a b a b a b a b a b c c c c c c d", 2],
]

LIVE = [
    [["Hello there.", *["Thank you."] * 6, "Thank you. Bye."], 4],
    [["so we went the the", "the the the the the the end", "the end"], 4],
    [["the the the the the the"] * 3, 0],
    [["one two", "one two", "one two", "one two", "one two three"], 3],
    [["谢谢", "谢谢谢谢谢谢谢谢", "谢谢你"], 4],
]


def main() -> None:
    seen: set[str] = set()
    detect_cases = []
    tagged = [(c, "server-test") for c in server_test_texts()] + [({"text": t}, "corpus") for t in CORPUS]
    for case, source in tagged:
        text = str(case["text"])
        if text in seen:
            continue
        seen.add(text)
        detect_cases.append({**case, "source": source, **detection(text)})
    trim_cases = []
    for text, limit in TRIM:
        seg, removed = trim_repeats(Segment(0.0, 30.0, str(text)), int(limit))
        trim_cases.append({"text": text, "max_repeats": limit, "out": seg.text, "removed": removed})
    live_cases = []
    for finals, limit in LIVE:
        guard = RepeatGuard(int(limit))
        out = [guard.apply(t)[0] for t in finals]
        live_cases.append({"finals": finals, "max_repeats": limit, "out": out, "removed": guard.removed})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"langdetect": detect_cases, "trim": trim_cases, "live": live_cases}, ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
    )
    export_features()
    print(f"{OUT}: {len(detect_cases)} texts, {len(trim_cases)} trims, {len(live_cases)} live sequences")


if __name__ == "__main__":
    main()
