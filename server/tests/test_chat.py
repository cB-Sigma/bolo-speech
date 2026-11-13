"""``/v1/chat/completions`` as a speech front end, through the official ``openai`` SDK and a fake language model."""

from __future__ import annotations

import base64

import httpx
import openai
import pytest

from bolo.audio import decode_audio
from conftest import LiveServer, wav_bytes


@pytest.fixture
def client(agent: LiveServer) -> openai.OpenAI:
    return openai.OpenAI(base_url=agent.url + "/v1", api_key="x", max_retries=0)


def audio_part(seconds: float = 2.0) -> dict:
    return {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(wav_bytes(seconds)).decode(), "format": "wav"},
    }


def test_text_only_requests_are_proxied_unchanged(client: openai.OpenAI, agent: LiveServer) -> None:
    result = client.chat.completions.create(
        model="some-model-name",
        messages=[{"role": "user", "content": "hello there"}],
        temperature=0.2,
        user="u1",
        extra_body={"top_k": 5},
    )
    assert result.choices[0].message.content == "You said: hello there. That is all I heard. Goodbye for now."
    assert result.usage is not None and result.usage.prompt_tokens == 7
    sent = agent.llm.requests[-1]  # type: ignore[attr-defined]
    assert (
        sent["model"] == "some-model-name"
        and sent["temperature"] == 0.2
        and sent["user"] == "u1"
        and sent["top_k"] == 5
    )
    assert sent["_authorization"] == "Bearer llm-secret"  # the operator's key, not the caller's
    stream = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text == "You said: hi. That is all I heard. Goodbye for now."
    assert not [t for t in httpx.get(agent.url + "/api/v1/requests").json()["requests"] if t["kind"] == "agent"]


def test_audio_input_is_transcribed_and_forwarded_as_text(client: openai.OpenAI, agent: LiveServer) -> None:
    result = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Listen:"}, audio_part(2.0)]},
        ],
        max_tokens=50,
    )
    assert (
        result.choices[0].message.content
        == "You said: Listen: the quick brown fox jumps. That is all I heard. Goodbye for now."
    )
    sent = agent.llm.requests[-1]  # type: ignore[attr-defined]
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Listen: the quick brown fox jumps"},
    ]
    assert sent["model"] == "fake-llm" and sent["max_tokens"] == 50 and "modalities" not in sent
    trace = httpx.get(agent.url + "/api/v1/requests", params={"kind": "agent"}).json()["requests"][0]
    assert (
        trace["transport"] == "chat"
        and trace["status"] == "ok"
        and trace["audio_seconds"] == 2.0
        and trace["audio_inputs"] == 1
    )
    assert {"decode", "llm"} <= {p["name"] for p in trace["phases"]} and trace["llm_model"] == "fake-llm"


def test_audio_output(client: openai.OpenAI, agent: LiveServer) -> None:
    result = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[{"role": "user", "content": "hello"}],
    )
    message = result.choices[0].message
    assert message.content is None and message.audio is not None
    assert message.audio.transcript == "You said: hello. That is all I heard. Goodbye for now."
    assert message.audio.id.startswith("audio_") and message.audio.expires_at > 1_700_000_000
    samples, rate = decode_audio(base64.b64decode(message.audio.data), filename="a.wav")
    assert rate == 24000 and len(samples) / rate == pytest.approx(len(message.audio.transcript) * 0.06, rel=0.15)
    sent = agent.llm.requests[-1]  # type: ignore[attr-defined]
    assert "audio" not in sent and "modalities" not in sent
    mp3 = client.chat.completions.create(
        model="x",
        modalities=["text", "audio"],
        audio={"voice": "beta", "format": "mp3"},
        messages=[{"role": "user", "content": [audio_part(1.0)]}],
    )
    assert base64.b64decode(mp3.choices[0].message.audio.data)[:2] in (b"\xff\xf3", b"\xff\xfb", b"ID")  # type: ignore[union-attr]


def test_streaming_audio_output_starts_before_the_reply_is_complete(client: openai.OpenAI, agent: LiveServer) -> None:
    agent.llm.token_delay_s = 0.04  # type: ignore[attr-defined]
    stream = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "pcm16"},
        messages=[{"role": "user", "content": [audio_part(2.0)]}],
        stream=True,
    )
    order: list[str] = []
    transcript, pcm = "", b""
    finish = None
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        finish = chunk.choices[0].finish_reason or finish
        audio = delta.audio  # typed by the SDK as ChoiceDeltaAudio
        if audio is not None and audio.transcript:
            transcript += audio.transcript
            order.append("t")
        if audio is not None and audio.data:
            pcm += base64.b64decode(audio.data)
            order.append("a")
    assert transcript == "You said: the quick brown fox jumps. That is all I heard. Goodbye for now."
    assert finish == "stop"
    assert len(pcm) / 2 / 24000 == pytest.approx(len(transcript) * 0.06, rel=0.15)
    first_audio, last_text = order.index("a"), len(order) - 1 - order[::-1].index("t")
    assert first_audio < last_text  # speech began while the model was still writing
    trace = httpx.get(agent.url + "/api/v1/requests", params={"kind": "agent"}).json()["requests"][0]
    assert trace["ttfa_ms"] is not None and trace["first_token_ms"] is not None and trace["ttfa_ms"] < trace["total_ms"]
    assert trace["audio_seconds_out"] > 2 and "llm_first_token" in [p["name"] for p in trace["phases"]]


def test_chat_errors(client: openai.OpenAI, agent: LiveServer, tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(openai.BadRequestError) as bad:
        client.chat.completions.create(
            model="x",
            messages=[
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "!!!", "format": "wav"}}]}
            ],
        )
    assert "base64" in str(bad.value) and bad.value.body["param"].startswith("messages[0].content[0]")  # type: ignore[index]
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model="x", modalities=["text", "audio"], messages=[{"role": "user", "content": "hi"}]
        )
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model="x",
            modalities=["text", "audio"],
            audio={"voice": "nobody", "format": "wav"},
            messages=[{"role": "user", "content": "hi"}],
        )
    agent.llm.fail_with = 500  # type: ignore[attr-defined]
    with pytest.raises(openai.APIStatusError) as upstream:
        client.chat.completions.create(model="x", messages=[{"role": "user", "content": [audio_part(1.0)]}])
    assert upstream.value.status_code == 502 and "the fake model is unwell" in str(upstream.value)
    proxied = httpx.post(
        agent.url + "/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert proxied.status_code == 500  # plain proxying hands back the upstream answer as it is
    agent.llm.fail_with = None  # type: ignore[attr-defined]
    # no language model configured
    from bolo.config import Settings

    live = LiveServer(
        Settings(
            data_dir=tmp_path / "d2", fake_engine=True, default_stt="fake-stt", default_tts="fake-tts", auto_pull=False
        )
    )
    live.start()
    unconfigured = openai.OpenAI(base_url=live.url + "/v1", api_key="x", max_retries=0)
    with pytest.raises(openai.APIStatusError) as none:
        unconfigured.chat.completions.create(model="x", messages=[{"role": "user", "content": "hi"}])
    assert none.value.status_code == 503 and none.value.body["code"] == "llm_not_configured"  # type: ignore[index]
    assert "llm_base_url" in str(none.value)
    assert httpx.post(agent.url + "/v1/chat/completions", json={"model": "x"}).status_code == 400
    caps = httpx.get(agent.url + "/api/v1/info").json()["capabilities"]
    assert caps["llm"] == {"configured": True, "base_url": agent.llm.url, "model": "fake-llm"}  # type: ignore[attr-defined]
    assert caps["chat_completions"]["audio"] is True and caps["realtime"]["conversation_sessions"] is True
    assert httpx.get(live.url + "/api/v1/info").json()["capabilities"]["realtime"]["conversation_sessions"] is False
    live.stop()


def test_llm_settings_and_doctor(
    agent: LiveServer, tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    from bolo.cli import main
    from bolo.config import load_settings

    cfg = tmp_path / "bolo.toml"
    cfg.write_text(
        '[llm]\nbase_url = "http://x:1/v1"\nmodel = "llama3.2"\napi_key = "k"\n\n[server]\napi_key = "server-key"\n'
    )
    s = load_settings(cfg, env={})
    assert (s.llm_base_url, s.llm_model, s.llm_api_key, s.api_key) == ("http://x:1/v1", "llama3.2", "k", "server-key")
    assert load_settings(env={"BOLO_LLM_BASE_URL": "http://y/v1"}).llm_base_url == "http://y/v1"
    assert load_settings(env={"CHAT_COMPLETION_BASE_URL": "http://z/v1"}).llm_base_url == "http://z/v1"  # speaches name
    assert s.public_dict()["llm_api_key"] == "***"
    monkeypatch.setenv("BOLO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOLO_LLM_BASE_URL", agent.llm.url)  # type: ignore[attr-defined]
    monkeypatch.setenv("BOLO_LLM_MODEL", "fake-llm")
    assert main(["doctor"]) == 0
    assert "reachable; model fake-llm: found" in capsys.readouterr().out
    monkeypatch.setenv("BOLO_LLM_MODEL", "other")
    main(["doctor"])
    assert "NOT in the endpoint's model list" in capsys.readouterr().out
    monkeypatch.setenv("BOLO_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    main(["doctor"])
    assert "NOT reachable" in capsys.readouterr().out
    monkeypatch.delenv("BOLO_LLM_BASE_URL")
    main(["doctor"])
    assert "language model: not configured" in capsys.readouterr().out
