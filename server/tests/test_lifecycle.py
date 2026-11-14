from __future__ import annotations

import asyncio
import time

import pytest

from bolo.catalog import ModelSpec
from bolo.config import Settings
from bolo.core import Bolo
from bolo.engines.fake import FakeSTT
from bolo.errors import EngineUnavailable, ModelNotFound, ModelNotInstalled, Overloaded


def add_fake(core: Bolo, model_id: str, **options: object) -> None:
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": model_id, "task": "stt", "engine": "fake", "kind": "fake-stt", "family": "Fake", "languages": ["en"],
                "license": "Apache-2.0", "builtin": True, "size_bytes": 100 * 1024 * 1024, "options": options,
            }
        )
    )  # fmt: skip


async def test_lazy_load_and_single_flight(core: Bolo) -> None:
    add_fake(core, "slow", load_delay_s=0.15)
    assert core.manager.loaded == {}
    events: list[dict] = []
    sub = core.events.subscribe()
    task = asyncio.ensure_future(sub.__anext__())
    await asyncio.sleep(0)
    t0 = time.perf_counter()
    entries = await asyncio.gather(*[core.manager.load("slow") for _ in range(5)])
    assert time.perf_counter() - t0 < 0.3  # five callers, one load
    assert all(e is entries[0] for e in entries)
    assert len(core.engines.get("fake").loaded) == 1
    assert entries[0].load_ms == pytest.approx(150, abs=60)
    events.append(await task)
    assert events[0] == {"type": "model", "id": "slow", "event": "loaded", "processor": "100% CPU"}
    await sub.aclose()


async def test_unknown_model_and_alias_resolution(core: Bolo) -> None:
    with pytest.raises(ModelNotFound):
        await core.manager.load("does-not-exist")
    entry = await core.manager.load("fake-realtime")  # alias
    assert entry.id == "fake-stt-streaming"
    assert core.resolve("whisper-1", "stt").id == "fake-stt"
    assert core.resolve("tts-1-hd", "tts").id == "fake-tts"
    assert core.resolve(None, "stt").id == "fake-stt"
    assert core.resolve("realtime", "stt").id == "fake-stt-streaming"
    with pytest.raises(Exception, match="cannot be used for tts"):
        core.resolve("fake-stt", "tts")


async def test_ttl_unloads_idle_models_but_not_busy_ones(settings: Settings) -> None:
    settings.stt_ttl_s = 0.15
    core = Bolo(settings)
    await core.start()
    try:
        async with core.manager.use("fake-stt") as entry:
            await asyncio.sleep(0.4)
            assert "fake-stt" in core.manager.loaded  # in use: kept despite the TTL
            assert entry.active == 1 and entry.idle_s() == 0.0
        await asyncio.sleep(0.05)
        assert "fake-stt" in core.manager.loaded
        await asyncio.sleep(0.4)
        assert "fake-stt" not in core.manager.loaded  # idle past the TTL
        await core.manager.load("fake-vad")  # vad_ttl_s = -1: never unloaded
        await asyncio.sleep(0.3)
        assert "fake-vad" in core.manager.loaded
    finally:
        await core.stop()


async def test_lru_eviction_by_model_count(settings: Settings) -> None:
    settings.max_loaded_models = 2
    core = Bolo(settings)
    await core.start()
    try:
        for name in ("a", "b", "c"):
            add_fake(core, name)
        await core.manager.load("a")
        await asyncio.sleep(0.01)
        await core.manager.load("b")
        async with core.manager.use("a"):  # touch a: b is now least recently used
            pass
        await core.manager.load("c")
        assert set(core.manager.loaded) == {"a", "c"}
        assert core.manager.evictions == 1
    finally:
        await core.stop()


async def test_memory_budget_evicts_idle_and_refuses_when_all_busy(settings: Settings) -> None:
    settings.memory_budget_mb = 250  # each fake model declares 100 MB (x1.2 headroom)
    core = Bolo(settings)
    await core.start()
    try:
        for name in ("a", "b", "c"):
            add_fake(core, name)
        a = await core.manager.load("a")
        b = await core.manager.load("b")
        a.memory_bytes = b.memory_bytes = 100 * 1024 * 1024
        assert core.manager.memory_in_use() == 200 * 1024 * 1024
        await core.manager.load("c")  # 200 + 120 > 250: the oldest idle model goes
        assert "a" not in core.manager.loaded and {"b", "c"} <= set(core.manager.loaded)
        core.manager.loaded["c"].memory_bytes = 100 * 1024 * 1024
        async with core.manager.use("b"), core.manager.use("c"):
            with pytest.raises(Overloaded, match="memory budget"):
                await core.manager.load("a")
    finally:
        await core.stop()


async def test_not_installed_and_engine_unavailable(core: Bolo) -> None:
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": "ghost", "task": "stt", "engine": "fake", "kind": "fake-stt", "family": "Fake", "languages": ["en"],
                "license": "MIT", "source": {"repo": "nobody/ghost"},
            }
        )
    )  # fmt: skip
    with pytest.raises(ModelNotInstalled, match="auto_pull is off"):
        await core.manager.load("ghost")
    core.catalog.add(
        ModelSpec.from_dict(
            {"id": "alien", "task": "stt", "engine": "no-such-engine", "kind": "x", "family": "X", "languages": [],
             "license": "MIT", "builtin": True}
        )
    )  # fmt: skip
    with pytest.raises(EngineUnavailable):
        await core.manager.load("alien")


async def test_gated_models_are_never_pulled_automatically(settings: Settings) -> None:
    settings.auto_pull = True
    core = Bolo(settings)
    await core.start()
    try:
        core.catalog.add(
            ModelSpec.from_dict(
                {
                    "id": "locked", "task": "stt", "engine": "fake", "kind": "fake-stt", "family": "Fake",
                    "languages": ["hi"], "license": "MIT", "gated": True, "source": {"repo": "org/locked"},
                    "source_url": "https://huggingface.co/org/locked",
                }
            )
        )  # fmt: skip
        with pytest.raises(Exception, match="gated") as exc:
            await core.manager.load("locked")
        assert exc.value.status == 401  # type: ignore[attr-defined]
        job = core.start_pull("locked")
        assert job.state == "error" and "BOLO_HF_TOKEN" in (job.error or "")
    finally:
        await core.stop()


async def test_unload_closes_queue_and_use_counts_requests(core: Bolo) -> None:
    async with core.manager.use("fake-stt") as entry:
        assert isinstance(entry.model, FakeSTT)
    assert entry.requests == 1 and entry.active == 0
    assert await core.manager.unload("fake-stt") is True
    assert await core.manager.unload("fake-stt") is False
    with pytest.raises(Overloaded):
        await entry.queue.submit("batch", None)
