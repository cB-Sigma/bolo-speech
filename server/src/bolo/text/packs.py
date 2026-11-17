"""uccharan's language packs, as bolo uses them.

uccharan (0.6) carries American English and Hindi in its wheel. Every other language's word table, British
English, the Kokoro English dictionaries and the optional share-alike data are **packs**: ``.ucd`` files attached
to uccharan's GitHub release, listed with their SHA-256 and size in ``uccharan.packs.PACKS``. uccharan reads them
from ``$UCCHARAN_DATA_DIR``, then from its cache (``~/.cache/uccharan/<version>/``).

bolo is the one that downloads them, so that a download is visible and can be refused:

* A voice whose language needs a pack that is not on this machine gets it on first use (:func:`ensure`): the
  download shows up as a ``pull`` job and in the log, is checked against the SHA-256 in the installed uccharan,
  goes into uccharan's cache, and is then loaded through ``uccharan.load_pack``, which checks it again.
* ``uccharan_auto_download = false`` (or ``UCCHARAN_OFFLINE=1`` in bolo's environment) turns that off: the request
  fails with 424 ``component_missing``, naming the pack and the command that fetches it.
* The CC BY-SA 4.0 packs (``wikipron-*``, ``xlit-*``) are never downloaded on their own, only by an explicit
  ``bolo pull uccharan-<pack>`` / ``bolo components install uccharan-<pack> --accept-license``.
* Every pack is a component, ``uccharan-<pack>`` (``uccharan-de``, ``uccharan-xlit-hi``).

While bolo runs, ``UCCHARAN_OFFLINE=1`` is set in its process so that uccharan itself never starts a download
behind bolo's back (it would otherwise fetch a missing pack silently on the first call).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from bolo.errors import ComponentMissing

log = logging.getLogger("bolo.packs")

#: component ids: ``uccharan-de`` is uccharan's pack ``de``
COMPONENT_PREFIX = "uccharan-"
APACHE = "Apache-2.0"
SHARE_ALIKE = "CC BY-SA 4.0"
#: packs under a share-alike licence: downloaded only when someone asks for them
SHARE_ALIKE_PREFIXES = ("wikipron-", "xlit-")
URL = "https://github.com/cB-Sigma/uccharan/releases/download/v{version}/{name}.ucd"
LICENSE_URL = "https://github.com/cb-Sigma/uccharan/blob/main/DATA_LICENSES.md"

#: uccharan language tags written in Latin script: Latin words in them are read by the language itself; in every
#: other language uccharan reads them with its British English data (the ``en-gb`` pack)
LATIN_SCRIPT = frozenset(
    "en-us en-gb es de nl fr fr-be fr-ch it pt-br pt pl tr cs sv ca ro hu fi nb da".split()  # noqa: SIM905
)
#: languages whose pack is another language's word table
SHARED_TABLE = {"fr-be": "fr", "fr-ch": "fr"}
#: packs that are part of the wheel's own data (listed in PACKS, never needed)
BUILT_IN = frozenset(("en-us", "hi"))
#: the uccharan languages that read with a built-in pack's data: Bhojpuri, Maithili and Magahi use Hindi's
BUILT_IN_USERS = {"en-us": frozenset(("en-us",)), "hi": frozenset(("hi", "bho", "mai", "mag"))}

LANGUAGE_NAMES = {
    "ca": "Catalan", "cs": "Czech", "da": "Danish", "de": "German", "en-gb": "British English",
    "en-us": "American English", "fi": "Finnish", "fr": "French", "hi": "Hindi", "hu": "Hungarian",
    "it": "Italian", "nb": "Norwegian Bokmål", "nl": "Dutch", "pl": "Polish", "pt": "European Portuguese",
    "pt-br": "Brazilian Portuguese", "ro": "Romanian", "sd": "Sindhi", "sv": "Swedish", "tr": "Turkish",
    "ur": "Urdu", "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu",
}  # fmt: skip


class PackMissing(ComponentMissing):
    """A language pack a voice needs is not on this machine and cannot (or may not) be downloaded."""


@dataclass(frozen=True)
class Pack:
    name: str
    sha256: str
    size: int

    @property
    def component(self) -> str:
        return COMPONENT_PREFIX + self.name

    @property
    def share_alike(self) -> bool:
        return is_share_alike(self.name)

    @property
    def license(self) -> str:
        return SHARE_ALIKE if self.share_alike else APACHE

    @property
    def language(self) -> str:
        """The language the pack is for (``xlit-hi`` -> ``hi``, ``kokoro-en-us`` -> ``en-us``)."""
        for prefix in ("xlit-", "wikipron-", "kokoro-"):
            if self.name.startswith(prefix):
                return self.name[len(prefix) :]
        return self.name

    @property
    def description(self) -> str:
        lang = LANGUAGE_NAMES.get(self.language, self.language)
        if self.name.startswith("xlit-"):
            return (
                f"Transliteration model and lexicon for the `romanised` option ({lang} typed in Latin letters, "
                "written in its own script). Built from share-alike data: downloaded only when you install it."
            )
        if self.name.startswith("wikipron-"):
            return (
                f"Wiktionary pronunciations ({lang}, via WikiPron) that uccharan's improved mode uses for words "
                "outside its own lexicon. Share-alike licence: downloaded only when you install it."
            )
        if self.name.startswith("kokoro-"):
            return f"misaki 0.9.4's {lang} dictionaries, for Kokoro voices in {lang} (uccharan's kokoro profile)."
        if self.name in BUILT_IN:
            return f"{lang}: part of the uccharan wheel; the pack is the same data as a separate file."
        return (
            f"{lang} word table (letter-to-sound model and exceptions) for phonemes and text normalisation. "
            "Downloaded the first time a voice in the language needs it."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "component": self.component,
            "size_bytes": self.size,
            "sha256": self.sha256,
            "license": self.license,
            "share_alike": self.share_alike,
            "language": self.language,
        }


def is_share_alike(name: str) -> bool:
    return name.startswith(SHARE_ALIKE_PREFIXES)


def component_id(name: str) -> str:
    return COMPONENT_PREFIX + name


def pack_of(component: str) -> str | None:
    """The pack a component id names (``uccharan-de`` -> ``de``), or None when it names no known pack."""
    if not component.startswith(COMPONENT_PREFIX):
        return None
    name = component[len(COMPONENT_PREFIX) :]
    return name if name in table() else None


def human_size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1_000_000 else f"{max(1, round(n / 1e3))} KB"


# -- the installed uccharan --------------------------------------------------------------------------------------


def module() -> Any | None:
    """The uccharan module (None when it is not installed, or hidden with ``BOLO_UCCHARAN=off``)."""
    if os.environ.get("BOLO_UCCHARAN", "auto").lower() == "off":
        return None
    try:
        import uccharan  # type: ignore[import-not-found,unused-ignore]
    except Exception:
        return None
    return uccharan


def _packs_module(mod: Any) -> Any | None:
    packs = getattr(mod, "packs", None)
    if packs is None:
        try:
            import importlib

            packs = importlib.import_module(f"{mod.__name__}.packs")
        except Exception:
            return None
    return packs


def table(mod: Any | None = None) -> dict[str, tuple[str, int]]:
    """``uccharan.packs.PACKS`` (name -> (sha256, size)); empty without uccharan or with one that has no packs."""
    mod = mod if mod is not None else module()
    if mod is None:
        return {}
    found = getattr(_packs_module(mod), "PACKS", None)
    return dict(found) if isinstance(found, dict) else {}


def all_packs() -> list[Pack]:
    return [Pack(name, str(sha), int(size)) for name, (sha, size) in sorted(table().items())]


def info(name: str) -> Pack | None:
    entry = table().get(name)
    return Pack(name, str(entry[0]), int(entry[1])) if entry else None


def version(mod: Any | None = None) -> str:
    mod = mod if mod is not None else module()
    return str(getattr(mod, "__version__", "") or "")


def url(name: str) -> str:
    return URL.format(version=version(), name=name)


def cache_dir() -> Path | None:
    packs = _packs_module(module()) if module() is not None else None
    fn = getattr(packs, "cache_dir", None)
    if not callable(fn):
        return None
    try:
        return Path(fn())
    except Exception:  # an unreadable home directory
        return None


def data_dirs() -> list[Path]:
    return [Path(d) for d in os.environ.get("UCCHARAN_DATA_DIR", "").split(os.pathsep) if d]


def command(name: str) -> str:
    """bolo's command that fetches the pack."""
    return f"bolo pull {component_id(name)}" + (" --accept-license" if is_share_alike(name) else "")


def manual_command(name: str) -> str:
    """uccharan's own command (no bolo needed)."""
    return f"python -m uccharan.packs {name}"


def offline_hint(name: str) -> str:
    return (
        f"run `{command(name)}` (or `{manual_command(name)}`); on a machine without network, fetch it elsewhere "
        f"with `python -m uccharan.packs --dir DIR {name}` (or download {url(name)}) and name that directory in "
        "UCCHARAN_DATA_DIR"
    )


# -- state -------------------------------------------------------------------------------------------------------

_lock = threading.Lock()
_pack_locks: dict[str, threading.Lock] = {}
#: packs this process handed to ``uccharan.load_pack``
_loaded: set[str] = set()
#: packs whose verified file was found (a positive answer does not change while bolo runs)
_present: set[str] = set()
_auto_download = True
#: UCCHARAN_OFFLINE as it was before bolo set it
_user_offline: bool | None = None


class Reporter(Protocol):
    """Who shows a download (the server: a ``pull`` job, events and the log)."""

    def pack_download(self, pack: Pack, reason: str, run: Callable[[Callable[[int], None]], None]) -> None: ...


_reporter: Reporter | None = None


def configure(auto_download: bool = True, reporter: Reporter | None = None) -> None:
    """Called by the server at start: the ``uccharan_auto_download`` setting, and who reports downloads."""
    global _auto_download, _reporter, _user_offline
    with _lock:
        if _user_offline is None:
            _user_offline = os.environ.get("UCCHARAN_OFFLINE") == "1"
        _auto_download = bool(auto_download)
        _reporter = reporter
        if table():
            os.environ["UCCHARAN_OFFLINE"] = "1"  # bolo downloads; uccharan only reads


def auto_download() -> bool:
    """Whether a missing Apache-2.0 pack is downloaded on first use."""
    return _auto_download and not _user_offline


def reset() -> None:
    """Forget what was loaded (tests, and after uccharan itself was installed or replaced)."""
    with _lock:
        _loaded.clear()
        _present.clear()


def _pack_lock(name: str) -> threading.Lock:
    with _lock:
        return _pack_locks.setdefault(name, threading.Lock())


def installed(name: str) -> bool:
    """Whether a verified copy of the pack is on this machine (data directory or cache), or it is loaded."""
    if name in _loaded or name in _present:
        return True
    mod = module()
    packs = _packs_module(mod) if mod is not None else None
    fn = getattr(packs, "installed", None)
    if not callable(fn):
        return False
    try:
        ok = bool(fn(name))
    except Exception:
        return False
    if ok:
        _present.add(name)
    return ok


def static_needs(language: str, profile: str = "espeak-ipa") -> list[str]:
    """The packs a uccharan language needs with a profile, worked out without asking uccharan (for listings)."""
    known = table()
    out: list[str] = []
    base = SHARED_TABLE.get(language, language)
    if base in known and base not in BUILT_IN:
        out.append(base)
    if profile == "kokoro" and f"kokoro-{language}" in known:
        out.append(f"kokoro-{language}")
    return out


_LATIN = re.compile(r"[A-Za-z]")


def needed(language: str, profile: str = "espeak-ipa", text: str = "") -> list[str]:
    """The packs uccharan still needs for ``language`` with ``profile`` (and ``text``), not yet loaded.

    The language's own packs come from uccharan (``missing_packs``). Latin-script words in a language written in
    another script are read with the British English data, so such text also needs ``en-gb``."""
    mod = module()
    if mod is None or not table(mod):
        return []
    prof = "kokoro" if profile == "kokoro" else "espeak-ipa"
    missing = getattr(getattr(mod, "_native", None), "missing_packs", None)
    names: list[str]
    if callable(missing):
        try:
            names = [str(n) for n in missing(language, prof)]
        except Exception:  # a tag this uccharan does not know: nothing it could load
            names = []
    else:
        names = static_needs(language, prof)
    if language not in LATIN_SCRIPT and text and _LATIN.search(text):
        names.append("en-gb")
    out: list[str] = []
    for n in names:
        if n not in out and n not in _loaded:
            out.append(n)
    return out


def load(name: str) -> None:
    """Hand a pack that is on this machine to uccharan (it checks the SHA-256 once more)."""
    mod = module()
    if mod is None:
        return
    try:
        mod.load_pack(name)
    except Exception as exc:
        raise PackMissing(f"uccharan could not load its language pack '{name}': {exc}") from exc
    with _lock:
        _loaded.add(name)
        _present.add(name)


def download(name: str, on_bytes: Callable[[int], None] | None = None) -> Path:
    """Fetch a pack from uccharan's release into its cache, checked against the SHA-256 and size the installed
    uccharan lists for it. Raises ``PackMissing`` (with the pack's name and how to get it) on any failure."""
    import httpx

    pack = info(name)
    if pack is None:
        raise PackMissing(f"the installed uccharan {version()} has no language pack '{name}'")
    target_dir = cache_dir()
    if target_dir is None:
        raise PackMissing(f"uccharan has no cache directory for its language pack '{name}': {offline_hint(name)}")
    source = url(name)
    digest = hashlib.sha256()
    blob = bytearray()
    try:
        with httpx.stream("GET", source, follow_redirects=True, timeout=60.0) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes(1 << 16):
                blob += chunk
                digest.update(chunk)
                if on_bytes is not None:
                    on_bytes(len(chunk))
                if len(blob) > pack.size:
                    break
    except (httpx.HTTPError, OSError) as exc:
        raise PackMissing(
            f"could not download uccharan's language pack '{name}' from {source} ({type(exc).__name__}: {exc}); "
            + offline_hint(name)
        ) from exc
    if len(blob) != pack.size or digest.hexdigest() != pack.sha256:
        raise PackMissing(
            f"uccharan's language pack '{name}' from {source} does not match the SHA-256 the installed uccharan "
            f"{version()} lists for it (got {digest.hexdigest()}, {len(blob)} bytes); it was not used"
        )
    target = target_dir / f"{name}.ucd"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as tmp:
            tmp.write(bytes(blob))
        Path(tmp.name).replace(target)
    except OSError as exc:
        raise PackMissing(f"could not write uccharan's language pack '{name}' to {target_dir}: {exc}") from exc
    return target


def removable(name: str) -> str | None:
    """Why the pack cannot be removed (None: it can): only a downloaded copy in uccharan's cache that this
    process has not loaded is removed; built-in packs and copies in ``$UCCHARAN_DATA_DIR`` stay."""
    if name in BUILT_IN:
        return "it is part of uccharan itself"
    if name in _loaded:
        return "this server has loaded it; restart the server first"
    folder = cache_dir()
    if folder is None or not (folder / f"{name}.ucd").is_file():
        return "it is not in uccharan's cache (a copy in UCCHARAN_DATA_DIR is managed by the operator)"
    return None


def remove(name: str) -> None:
    """Delete the pack's downloaded copy from uccharan's cache (see :func:`removable`)."""
    with _pack_lock(name):
        reason = removable(name)
        if reason is not None:
            raise ValueError(f"uccharan's pack '{name}' cannot be removed: {reason}")
        folder = cache_dir()
        assert folder is not None
        (folder / f"{name}.ucd").unlink(missing_ok=True)
        with _lock:
            _present.discard(name)


def install(name: str, on_bytes: Callable[[int], None] | None = None) -> None:
    """Download (unless it is already here) and load one pack: what an explicit pull does."""
    with _pack_lock(name):
        if not installed(name):
            download(name, on_bytes)
        load(name)


def _missing_message(name: str, language: str, why: str) -> str:
    pack = info(name)
    what = f"uccharan's language pack '{name}'"
    if pack is not None:
        what += f" ({human_size(pack.size)}, {pack.license})"
    return f"{what} is needed for language '{language}' and is not installed; {why}: {offline_hint(name)}"


def ensure(language: str, profile: str = "espeak-ipa", text: str = "", *, required: bool = True) -> list[str]:
    """Make sure uccharan has every pack it needs for ``language``: load the ones on this machine, download the
    missing Apache-2.0 ones when automatic downloads are on. Returns the packs downloaded.

    ``required`` (phonemes: uccharan never reads a language without its word table) raises ``PackMissing``
    (424) when a pack cannot be had; otherwise (normalisation, where uccharan falls back to its rules) the pack is
    left out and the caller decides what to do with the answer of :func:`needed`."""
    fetched: list[str] = []
    for name in needed(language, profile, text):
        with _pack_lock(name):
            if name in _loaded:
                continue
            if installed(name):
                load(name)
                continue
            if is_share_alike(name):
                if required:
                    raise PackMissing(_missing_message(name, language, "it is licensed CC BY-SA 4.0, so bolo never "
                                                       "downloads it on its own"))  # fmt: skip
                continue
            if not auto_download():
                if not required:
                    continue
                why = (
                    "UCCHARAN_OFFLINE=1 is set"
                    if _user_offline
                    else "automatic downloads are off (uccharan_auto_download = false)"
                )
                raise PackMissing(_missing_message(name, language, why))
            pack = info(name)
            if pack is None:
                if required:
                    raise PackMissing(f"the installed uccharan {version()} has no language pack '{name}'")
                continue
            reason = f"language '{language}'"

            def run(progress: Callable[[int], None], n: str = name) -> None:
                download(n, progress)

            try:
                if _reporter is not None:
                    _reporter.pack_download(pack, reason, run)
                else:
                    log.info("downloading uccharan language pack '%s' (%s) for %s", name, human_size(pack.size), reason)
                    download(name)
            except PackMissing:
                if required:
                    raise
                log.warning("uccharan language pack '%s' could not be downloaded; continuing without it", name)
                continue
            load(name)
            fetched.append(name)
    return fetched
