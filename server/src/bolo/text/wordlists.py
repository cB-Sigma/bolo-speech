"""Named pronunciation word lists, stored as ``<data_dir>/wordlists/<name>.tsv``.

The format is uccharan's (``word lang ipa say match note``, see uccharan's docs/wordlists.md);
PLS (W3C Pronunciation Lexicon) and JSON are accepted on upload and offered on export. Parsing
and matching need the ``uccharan`` package; without it lists can still be listed and deleted.

The list named ``default`` applies to every request that does not choose its own lists.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
from pathlib import Path
from typing import Any

from bolo.errors import BoloError, ComponentMissing, InvalidRequest

NAME = re.compile(r"^[a-z0-9_-]{1,64}$")
#: limits (one list, one request)
MAX_ENTRIES = 10_000
MAX_BYTES = 1_000_000
MAX_LISTS_PER_REQUEST = 8
MAX_INLINE_ENTRIES = 500
DEFAULT = "default"


class WordListNotFound(BoloError):
    status = 404
    code = "wordlist_not_found"
    type = "invalid_request_error"


def uccharan_module() -> Any | None:
    """The uccharan module when it has word-list support, else None."""
    try:
        import uccharan
    except Exception:
        return None
    return uccharan if hasattr(uccharan, "WordList") and hasattr(uccharan, "normalize") else None


def require_uccharan() -> Any:
    mod = uccharan_module()
    if mod is None:
        raise ComponentMissing(
            "pronunciation word lists need a uccharan release with word-list support: pip install -U uccharan"
        )
    return mod


def check_name(name: str) -> str:
    if not NAME.match(name):
        raise InvalidRequest("a word-list name uses a-z, 0-9, '-' and '_' (at most 64 characters)", param="name")
    return name


def parse(text: str, *, param: str = "body") -> Any:
    """Parse TSV, PLS or JSON into a ``uccharan.WordList`` (400 with the line on errors)."""
    mod = require_uccharan()
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise InvalidRequest(f"a word list is limited to {MAX_BYTES} bytes", param=param)
    try:
        wl = mod.WordList(text)
    except ValueError as exc:
        raise InvalidRequest(f"invalid word list: {exc}", param=param) from exc
    if len(wl) > MAX_ENTRIES:
        raise InvalidRequest(f"a word list is limited to {MAX_ENTRIES} entries", param=param)
    return wl


def inline(entries: list[dict[str, Any]], *, param: str = "pronunciations") -> Any:
    """A word list from request entries ``{"word", "ipa"?, "say"?, "lang"?, "match"?}``."""
    mod = require_uccharan()
    if len(entries) > MAX_INLINE_ENTRIES:
        raise InvalidRequest(f"at most {MAX_INLINE_ENTRIES} inline pronunciations", param=param)
    clean: list[dict[str, str]] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or not str(e.get("word", "")).strip():
            raise InvalidRequest(f"pronunciations[{i}] needs a word", param=param)
        if not e.get("ipa") and not e.get("say"):
            raise InvalidRequest(f"pronunciations[{i}] needs ipa or say", param=param)
        clean.append({k: str(v) for k, v in e.items() if k in ("word", "ipa", "say", "lang", "match", "note") and v})
    try:
        return mod.WordList.from_entries(clean)
    except ValueError as exc:
        raise InvalidRequest(f"invalid pronunciations: {exc}", param=param) from exc


class WordListStore:
    """Word lists on disk, parsed once per file change."""

    def __init__(self, data_dir: Path) -> None:
        self.dir = Path(data_dir) / "wordlists"
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[int, Any]] = {}

    def path(self, name: str) -> Path:
        return self.dir / f"{check_name(name)}.tsv"

    def names(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.tsv") if NAME.match(p.stem))

    def exists(self, name: str) -> bool:
        return NAME.match(name) is not None and self.path(name).is_file()

    def text(self, name: str) -> str:
        p = self.path(name)
        try:
            return p.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise WordListNotFound(f"no word list named '{name}'", param="name") from None

    def load(self, name: str) -> Any:
        """The parsed list (a ``uccharan.WordList``)."""
        p = self.path(name)
        try:
            stamp = p.stat().st_mtime_ns
        except FileNotFoundError:
            raise WordListNotFound(f"no word list named '{name}'", param="name") from None
        with self._lock:
            hit = self._cache.get(name)
            if hit is not None and hit[0] == stamp:
                return hit[1]
        wl = parse(self.text(name))
        with self._lock:
            self._cache[name] = (stamp, wl)
        return wl

    def summary(self, name: str) -> dict[str, Any]:
        p = self.path(name)
        try:
            st = p.stat()
        except FileNotFoundError:
            raise WordListNotFound(f"no word list named '{name}'", param="name") from None
        info: dict[str, Any] = {
            "name": name,
            "bytes": st.st_size,
            "updated": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "entries": None,
            "languages": [],
        }
        if uccharan_module() is not None:
            try:
                wl = self.load(name)
                info["entries"] = len(wl)
                info["languages"] = wl.languages()
            except BoloError as exc:
                info["error"] = exc.message
        return info

    def all(self) -> list[dict[str, Any]]:
        return [self.summary(n) for n in self.names()]

    def entries(self, name: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self.load(name).entries()
        return out

    def export(self, name: str, fmt: str, lang: str | None = None) -> str:
        if fmt == "tsv":
            return self.text(name)
        wl = self.load(name)
        if fmt == "pls":
            return str(wl.to_pls(lang))
        if fmt == "json":
            return json.dumps({"name": name, "entries": wl.entries()}, ensure_ascii=False)
        raise InvalidRequest("format must be tsv, pls or json", param="format")

    def put(self, name: str, body: str) -> dict[str, Any]:
        """Replace a list with ``body`` (TSV, PLS or JSON); stored as TSV."""
        check_name(name)
        wl = parse(body)
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".{name}.tsv.tmp"
        tmp.write_text(wl.to_tsv(), encoding="utf-8")
        tmp.replace(self.path(name))
        with self._lock:
            self._cache.pop(name, None)
        out = self.summary(name)
        out["warnings"] = [{"line": ln, "message": m} for ln, m in wl.warnings()]
        return out

    def delete(self, name: str) -> None:
        try:
            self.path(name).unlink()
        except FileNotFoundError:
            raise WordListNotFound(f"no word list named '{name}'", param="name") from None
        with self._lock:
            self._cache.pop(name, None)


_stores: dict[Path, WordListStore] = {}
_stores_lock = threading.Lock()


def store_for(data_dir: Path) -> WordListStore:
    """One store per data directory (shared by the API and synthesis)."""
    key = Path(data_dir)
    with _stores_lock:
        s = _stores.get(key)
        if s is None:
            s = _stores[key] = WordListStore(key)
        return s
