"""Language detection for text-to-speech: the detector, and how a request without a language uses it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bolo.api import create_app
from bolo.config import Settings
from bolo.text.langdetect import base, detect

# -- the detector ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "language", "script"),
    [
        ("नमस्ते, आप कैसे हैं? मैं ठीक हूँ।", "hi", "Devanagari"),
        ("मी घरी आहे आणि तुम्ही कुठे आहात?", "mr", "Devanagari"),
        ("म घर जान्छु। यो राम्रो छ।", "ne", "Devanagari"),
        ("मैं कल office जा रहा हूँ", "hi", "Devanagari"),  # English words in Latin letters do not decide
        ("வணக்கம், எப்படி இருக்கிறீர்கள்?", "ta", "Tamil"),
        ("আমি ভাত খাই", "bn", "Bengali"),
        ("মই ভাত খাওঁ, তুমি কি কৰা?", "as", "Bengali"),
        ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "pa", "Gurmukhi"),
        ("કેમ છો?", "gu", "Gujarati"),
        ("నమస్కారం", "te", "Telugu"),
        ("ನಮಸ್ಕಾರ", "kn", "Kannada"),
        ("നമസ്കാരം", "ml", "Malayalam"),
        ("ନମସ୍କାର", "or", "Oriya"),
        ("مرحبا كيف حالك", "ar", "Arabic"),
        ("آپ کیسے ہیں؟ میں ٹھیک ہوں", "ur", "Arabic"),
        ("Привет, как дела? Это хорошо.", "ru", "Cyrillic"),
        ("Привіт, як справи? Це добре.", "uk", "Cyrillic"),
        ("こんにちは、元気ですか", "ja", "Kana"),
        ("今日は東京へ行きます", "ja", "Kana"),
        ("你好，今天天气很好", "zh", "Han"),
        ("안녕하세요, 반갑습니다", "ko", "Hangul"),
        ("สวัสดีครับ", "th", "Thai"),
        ("Γεια σου, τι κάνεις;", "el", "Greek"),
        ("שלום, מה שלומך?", "he", "Hebrew"),
        ("Hello, how are you doing today? It is a good day.", "en", "Latin"),
        ("Hola, ¿cómo estás? Yo estoy muy bien, gracias.", "es", "Latin"),
        ("Bonjour, je suis très content de vous voir ici.", "fr", "Latin"),
        ("Guten Tag, ich bin nicht sicher, ob das stimmt.", "de", "Latin"),
        ("Ciao, come stai? Io sono molto contento di questo.", "it", "Latin"),
        ("Olá, você está bem? Não sei se ele vem para a festa.", "pt", "Latin"),
        ("Dit is een goed idee, maar ik heb er geen tijd voor.", "nl", "Latin"),
        ("Dzień dobry, nie wiem, czy to jest prawda.", "pl", "Latin"),
        ("Bu çok güzel bir gün ve ben de mutluyum.", "tr", "Latin"),
    ],
)
def test_detects_language_and_script(text: str, language: str, script: str) -> None:
    found = detect(text)
    assert (found.language, found.script) == (language, script), found
    assert found.sure and 0 < found.confidence <= 1


@pytest.mark.parametrize("text", ["OK", "Hello there", "12345", "", "Sí", "Taxi, hotel, radio, piano."])
def test_unsure_when_the_text_does_not_tell(text: str) -> None:
    assert detect(text).language is None


def test_markup_is_ignored_and_alternatives_are_given() -> None:
    assert detect('<speak>नमस्ते <break time="1s"/> आप कैसे हैं?</speak>').language == "hi"
    assert detect("[Kokoro](/kˈOkəɹO/) नमस्ते, आप कैसे हैं?").script == "Devanagari"
    found = detect("नमस्ते, आप कैसे हैं?")
    assert found.alternatives[:2] == ("mr", "ne")
    assert base("hi-IN") == "hi" and base("en_US") == "en" and base("nb") == "no" and base(None) == ""


# -- requests --------------------------------------------------------------------------------------------------


@pytest.fixture
def client(settings: Settings) -> TestClient:
    with TestClient(create_app(settings)) as c:
        yield c  # type: ignore[misc]


def speak(client: TestClient, **body: object) -> object:
    r = client.post("/v1/audio/speech", json={"model": "fake-tts", "response_format": "wav", **body})
    assert r.status_code == 200, r.text
    return r


def test_detected_language_picks_a_voice(client: TestClient) -> None:
    # fake-tts: alpha (en-US), beta (en-GB), gamma (hi-IN)
    r = speak(client, input="नमस्ते, आप कैसे हैं? मैं ठीक हूँ।")
    assert r.headers["x-bolo-language"] == "hi-IN; source=detected; detected=hi; script=Devanagari; voice=gamma"
    core = client.app.state.core  # type: ignore[attr-defined]
    trace = core.traces.recent(1)[0]
    assert trace.language == "hi-IN"
    assert trace.extra["language"]["voice"] == "gamma" and trace.extra["language"]["source"] == "detected"
    # Marathi: the model has no Marathi voice, so the closest language of the script (Hindi) answers
    r = speak(client, input="मी घरी आहे आणि तुम्ही कुठे आहात?")
    assert r.headers["x-bolo-language"] == (
        "hi-IN; source=detected; detected=mr; script=Devanagari; matched=hi; voice=gamma"
    )
    r = speak(client, input="Hello, how are you doing today? It is a good day.")
    assert r.headers["x-bolo-language"] == "en-US; source=detected; detected=en; script=Latin; voice=alpha"


def test_unsure_or_unoffered_keeps_the_default(client: TestClient) -> None:
    r = speak(client, input="OK.")
    assert r.headers["x-bolo-language"].startswith("en-US; source=default; detected=unsure")
    r = speak(client, input="OK.", voice="beta")
    assert r.headers["x-bolo-language"].startswith("en-GB; source=voice; detected=unsure")
    r = speak(client, input="Bonjour, je suis très content de vous voir ici.")
    assert r.headers["x-bolo-language"] == "en-US; source=default; detected=fr; script=Latin; offered=false"


def test_explicit_voice_and_language_are_never_overridden(client: TestClient) -> None:
    hindi = "नमस्ते, आप कैसे हैं? मैं ठीक हूँ।"
    r = speak(client, input=hindi, voice="alpha")
    assert r.headers["x-bolo-language"] == "en-US; source=voice; detected=hi; script=Devanagari; mismatch=true"
    r = speak(client, input=hindi, language="en")
    assert r.headers["x-bolo-language"] == "en; source=request"
    r = speak(client, input=hindi, detect_language=False)
    assert "x-bolo-language" not in r.headers
    core = client.app.state.core  # type: ignore[attr-defined]
    assert core.traces.recent(1)[0].language == "en-US"  # the default voice, as before
    # a stock client's OpenAI voice name is a placeholder: the voice that speaks the text's language answers
    r = speak(client, input=hindi, voice="alloy")
    assert r.headers["x-bolo-language"].endswith("voice=gamma")
    # a voice of the right language is kept as it is
    r = speak(client, input="Hello, how are you doing today? It is a good day.", voice="beta")
    assert r.headers["x-bolo-language"] == "en-GB; source=voice; detected=en; script=Latin"


def test_a_model_with_one_language_is_left_alone(client: TestClient) -> None:
    r = client.post(
        "/v1/audio/speech",
        json={"model": "fake-tts-clone", "input": "नमस्ते, आप कैसे हैं?", "response_format": "wav"},
    )
    assert r.status_code == 200 and r.headers["x-bolo-language"] == "en-US; source=model"


CLONE_MODEL = """\
- id: fake-tts-clone
  task: tts
  engine: fake
  kind: fake-tts
  family: Fake
  languages: [en]
  tier: realtime
  sample_rate: 24000
  license: Apache-2.0
  builtin: true
  options: {declared_voices: true}
  voices: [{id: alpha, name: Alpha, language: en-US, gender: f}]
  description: Deterministic test voice with one language.
- id: fake-tts-multi
  task: tts
  engine: fake
  kind: fake-tts
  family: Fake
  languages: [en, hi, ta]
  tier: realtime
  sample_rate: 24000
  license: Apache-2.0
  builtin: true
  options: {cloning: true, declared_voices: true}
  voices: [{id: alpha, name: Alpha, gender: f}]
  description: Deterministic test voice that is told the language.
"""


@pytest.fixture(autouse=True)
def custom_models(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "custom.yaml").write_text(CLONE_MODEL)


def test_a_voice_without_a_language_is_told_the_language(client: TestClient) -> None:
    r = client.post(
        "/v1/audio/speech",
        json={"model": "fake-tts-multi", "input": "வணக்கம், எப்படி இருக்கிறீர்கள்?", "response_format": "wav"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["x-bolo-language"] == "ta; source=detected; detected=ta; script=Tamil"
    core = client.app.state.core  # type: ignore[attr-defined]
    model = core.manager.loaded["fake-tts-multi"].model
    assert model.extras[-1].language == "ta"  # type: ignore[attr-defined]
