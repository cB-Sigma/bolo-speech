"""Text models that run on a transcript after recognition: inverse text normalisation (task ``itn``).

* ``wetext`` engine, kind ``wetext_itn``: WeTextProcessing's grammars for one language (a tagger and a verbaliser,
  weighted finite-state transducers, Apache-2.0) run by the ``wetext`` package (Apache-2.0) through kaldifst. The
  catalog entry pulls that language's grammars; the engine points wetext at them.
* ``kaldifst`` engine, kind ``rule_fst``: one rule FST applied as it is with kaldifst's ``TextNormalizer``
  (sherpa-onnx's ``itn_zh_number.fst``, the rule FSTs its recognisers take as ``rule_fsts``).

Both need ``pip install "bolo-speech[itn]"`` (wetext and kaldifst, Apache-2.0).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from bolo.engines.base import Engine, EngineStatus, ItnModel, LoadOptions, Model

INSTALL = 'pip install "bolo-speech[itn]"'
_wetext_lock = threading.Lock()


def _find(path: Path, name: str) -> Path:
    found = sorted(path.rglob(name))
    if not found:
        raise FileNotFoundError(f"no {name} under {path}")
    return found[0]


class WetextItn(ItnModel):
    """One language's WeTextProcessing inverse-normalisation grammars from the model directory."""

    def __init__(self, path: Path, spec: dict[str, Any]) -> None:
        import wetext
        from wetext import constants, token_parser

        self.language = str(spec.get("language") or (spec.get("languages") or ["en"])[0])
        grammars = _find(path, "tagger.fst").parent
        with _wetext_lock:
            # wetext reads its graphs from the paths in this table and the token order from its package: both are
            # pointed at the grammars this model pulled, for this language and direction only
            constants.FST_PATHS.setdefault(self.language, {})["itn"] = {
                fst.stem: str(fst) for fst in sorted(grammars.glob("*.fst"))
            }
            orders = grammars / "token_orders.json"
            if orders.is_file():
                original = getattr(token_parser.get_token_orders, "bolo_original", token_parser.get_token_orders)
                pulled = {(self.language, "itn"): json.loads(orders.read_text("utf-8"))}
                known = dict(getattr(token_parser.get_token_orders, "bolo_pulled", {}))
                known.update(pulled)

                def get_token_orders(lang: str, operator: str) -> Any:
                    return known[(lang, operator)] if (lang, operator) in known else original(lang, operator)

                get_token_orders.bolo_original = original  # type: ignore[attr-defined]
                get_token_orders.bolo_pulled = known  # type: ignore[attr-defined]
                token_parser.get_token_orders = get_token_orders  # type: ignore[assignment]
            self._normalizer = wetext.Normalizer(lang=self.language, operator="itn")
            self._normalizer.normalize("one")  # load the graphs now, not on the first request
        self._lock = threading.Lock()

    def normalize(self, text: str) -> str:
        with self._lock:
            return str(self._normalizer.normalize(text)).strip()


class RuleFstItn(ItnModel):
    """A single rule FST (input words to written form), applied with kaldifst's TextNormalizer."""

    def __init__(self, path: Path, spec: dict[str, Any]) -> None:
        import kaldifst

        self._normalizer = kaldifst.TextNormalizer(str(_find(path, str(spec.get("file", "*.fst")))))
        self._lock = threading.Lock()

    def normalize(self, text: str) -> str:
        with self._lock:
            return str(self._normalizer.normalize(text)).strip()


class WetextEngine(Engine):
    id = "wetext"
    name = "wetext (WeTextProcessing grammars)"
    license = "Apache-2.0"
    url = "https://github.com/pengzhendong/wetext"
    kinds = ("wetext_itn",)

    def status(self) -> EngineStatus:
        try:
            from importlib.metadata import version

            import wetext  # noqa: F401

            return EngineStatus(self.id, True, version("wetext"), None)
        except Exception as exc:
            return EngineStatus(self.id, False, None, f"{type(exc).__name__}: {exc}. Install with: {INSTALL}")

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind != "wetext_itn":
            raise ValueError(f"wetext engine cannot load kind {kind!r}")
        return WetextItn(path, spec)


class KaldiFstEngine(Engine):
    id = "kaldifst"
    name = "kaldifst (rule FSTs)"
    license = "Apache-2.0"
    url = "https://github.com/k2-fsa/kaldifst"
    kinds = ("rule_fst",)

    def status(self) -> EngineStatus:
        try:
            from importlib.metadata import version

            import kaldifst  # noqa: F401

            return EngineStatus(self.id, True, version("kaldifst"), None)
        except Exception as exc:
            return EngineStatus(self.id, False, None, f"{type(exc).__name__}: {exc}. Install with: {INSTALL}")

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind != "rule_fst":
            raise ValueError(f"kaldifst engine cannot load kind {kind!r}")
        return RuleFstItn(path, spec)
