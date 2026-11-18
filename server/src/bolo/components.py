"""Optional third-party components that bolo can use but does not require.

* **uccharan** (Apache-2.0): the phonemiser for Piper, Kitten, Matcha and Kokoro voices in the
  languages it covers (39 since uccharan 0.6: English, Hindi and the other Indian languages, Urdu, Sindhi and
  nineteen European languages, see bolo.text.phonemize). Part of ``bolo-speech[all]``; installable here.
* **uccharan's language packs** (``uccharan-<pack>``): the word tables of the languages that are not in uccharan's
  wheel, downloaded on first use (bolo.text.packs), and the CC BY-SA 4.0 packs, installed only on request.
* **eSpeak NG** (GPL-3.0-or-later): the phonemiser for every other language. Never shipped or
  installed by default: only when the operator asks for it (console, ``bolo components install
  espeak-ng``, the ``espeak`` extra, or the system package manager).
* **Engine plugins** (``plugin:<name>``): out-of-process engines such as a PyTorch engine
  (docs/plugins.md). Listed so that the console can say which plugin a model needs; they are
  installed as packages by the operator, not from here.
* **TTS server runtimes** (``llama-server`` from llama.cpp, MIT; ``audio.cpp``, Apache-2.0): the binaries the
  ``tts-server`` engine starts for its models (bolo.tts_runtimes). Installable here as pinned upstream release
  archives checked against their SHA-256, or provided by the operator (a setting, PATH, or a running server
  named in ``tts_servers``, in which case the model does not need the binary at all).
* **GPU builds of PyTorch** (``torch-cuda``, ``torch-rocm``, ``torch-xpu``, ``torch-cpu``): torch reinstalled from
  PyTorch's own index in every environment a bolo-torch plugin runs in (bolo.torch_envs), since PyPI's torch is
  CPU only on Windows and macOS. Listed for the GPU the machine has.
* **Full-duplex model servers** (``moshi``: Kyutai's PyTorch, MLX and Rust servers, MIT / Apache-2.0;
  ``personaplex``: NVIDIA's PersonaPlex server, MIT): the servers of Moshi, Hibiki and PersonaPlex
  (bolo.duplex.runtime), each build in a virtual environment (or cargo root) of its own, pinned by version and
  SHA-256 (a git commit for PersonaPlex). Several GB: PyTorch with CUDA.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bolo import torch_envs, tts_runtimes
from bolo.duplex import runtime as duplex_runtime
from bolo.errors import BoloError, InvalidRequest, ModelNotFound
from bolo.jobs import Job
from bolo.text import packs, phonemize, romanised

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo

ESPEAK = phonemize.ESPEAK
UCCHARAN = phonemize.UCCHARAN
ESPEAK_PACKAGE = "espeakng-loader>=0.2.4"
UCCHARAN_PACKAGE = phonemize.UCCHARAN_REQUIREMENT
PACKAGES = {ESPEAK: ESPEAK_PACKAGE, UCCHARAN: UCCHARAN_PACKAGE}
LOOPBACK = ("127.0.0.1", "localhost", "::1")
PLUGIN_PREFIX = "plugin:"
#: GPU builds of PyTorch for bolo-torch's environments: "torch-cuda" (bolo.torch_envs)
TORCH_PREFIX = "torch-"
TORCH_COMPONENTS = tuple(TORCH_PREFIX + t for t in torch_envs.TARGETS)
#: how long the bolo-torch environments and the GPU, as last looked at, are taken as they are (seconds)
TORCH_STATUS_TTL_S = 120.0
#: uccharan's romanised-text packs as components: "uccharan-xlit-hi" (bolo.text.romanised)
XLIT_PREFIX = romanised.COMPONENT_PREFIX + romanised.PACK_PREFIX
#: every other uccharan pack as a component: "uccharan-de" (bolo.text.packs)
PACK_PREFIX = packs.COMPONENT_PREFIX
#: Kokoro voice-name prefix -> phonemiser language (Mandarin "z" reads from its lexicon)
KOKORO_LANGUAGES = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br", "j": "ja"}

log = logging.getLogger(__name__)


def espeak_status() -> tuple[bool, str | None, str | None]:
    """``(installed, source, version)`` of the espeak-ng the phonemiser would use."""
    ph = phonemize.get_phonemizer()
    if ph is None:
        return False, None, None
    return True, ph.source, ph.version()


def uccharan_status() -> tuple[bool, str | None, str | None]:
    ph = phonemize.get_uccharan()
    if ph is None:
        return False, None, None
    return True, ph.source, ph.version()


def installer_command() -> list[str] | None:
    """The command that installs a package into the running environment, or None."""
    try:
        importlib.import_module("pip")
        return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    except ImportError:
        pass
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable]
    return None


def manual_command(component: str = ESPEAK) -> str:
    """How a person would install the component for the way bolo itself was installed."""
    extra, package = ("espeak", "espeakng-loader") if component == ESPEAK else ("uccharan", "uccharan")
    prefix = sys.prefix.replace("\\", "/")
    if Path("/.dockerenv").exists():
        if component == ESPEAK:
            return "rebuild the image with: docker build --build-arg WITH_ESPEAK=1 ..."
        return "rebuild the image (uccharan is part of bolo-speech[all])"
    if "/pipx/venvs/" in prefix:
        return f"pipx inject bolo-speech {package}"
    if "/uv/tools/" in prefix:
        return f"uv tool install --with {package} bolo-speech"
    return f'pip install "bolo-speech[{extra}]"'  # double quotes: cmd.exe does not take single ones


class ComponentManager:
    def __init__(self, core: Bolo) -> None:
        self.core = core
        #: replaced in tests; returns (exit code, output)
        self.run_installer: Callable[[list[str]], tuple[int, str]] = self._run
        #: the steps of installing a full-duplex server (uv / pip / git / cargo); replaced in tests
        self.run_duplex_step: Callable[[list[str]], tuple[int, str]] = duplex_runtime.run_command
        #: commands for the PyTorch builds (probes, pip, nvidia-smi) and PyTorch's index pages; replaced in tests
        self.torch_runner: torch_envs.Runner = torch_envs.run
        self.torch_page: Callable[[str, str], str] = torch_envs.fetch_page
        self._torch_cache: tuple[float, list[torch_envs.TorchEnv], torch_envs.Gpu] | None = None

    @staticmethod
    def _run(cmd: list[str]) -> tuple[int, str]:
        proc = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        return proc.returncode, (proc.stdout + proc.stderr).decode(errors="replace")

    # -- what a model needs -------------------------------------------------
    def pinned_phonemizer(self, spec: ModelSpec) -> str | None:
        value = self.core.store.model_settings(spec.id).get("phonemizer")
        return str(value) if value else None

    @staticmethod
    def phoneme_target(spec: ModelSpec, voice_id: str | None = None) -> tuple[str, str] | None:
        """``(profile, language)`` of a voice that reads phonemes, or None (characters or lexicon only)."""
        if spec.needs_phonemes:
            return spec.phoneme_target()
        if spec.kind == "kokoro" and spec.engine in ("sherpa-onnx", "mlx"):
            prefix = (voice_id or "a")[:1] if spec.options.get("multi_lang") else "a"
            language = KOKORO_LANGUAGES.get(prefix)
            return ("kokoro", language) if language else None
        return None

    def phonemizer_options(self, profile: str, language: str, pinned: str | None = None) -> list[dict[str, Any]]:
        """The phonemisers a voice can use: ``[{id, available, default, note}]`` (for a selector in the console)."""
        covered = phonemize.uccharan_language(profile, language) is not None
        uc, es = self.is_installed(UCCHARAN), self.is_installed(ESPEAK)
        supported = ", ".join(sorted(phonemize.UCCHARAN_SUPPORT.get(profile, {})))
        options: list[dict[str, Any]] = []
        if covered:
            options.append(
                {
                    "id": UCCHARAN,
                    "available": uc,
                    "default": False,
                    "note": "uccharan, compatible mode: the phonemes the voice was trained on (Apache-2.0)"
                    + ("" if uc else "; not installed"),
                }
            )
            options.append(
                {
                    "id": f"{UCCHARAN}:improved",
                    "available": uc,
                    "default": False,
                    "note": "uccharan, improved mode: better homographs, numbers and Hindi schwa deletion, "
                    "but not what the voice was trained on" + ("" if uc else "; not installed"),
                }
            )
        else:
            options.append(
                {
                    "id": UCCHARAN,
                    "available": False,
                    "default": False,
                    "note": f"uccharan does not cover '{language}' for {profile} voices (it covers {supported})",
                }
            )
        options.append(
            {
                "id": ESPEAK,
                "available": es,
                "default": False,
                "note": "eSpeak NG, optional component (GPL-3.0-or-later)" + ("" if es else "; not installed"),
            }
        )
        chosen: str | None = None
        if pinned:
            component, improved = phonemize.parse_choice(pinned)
            chosen = f"{UCCHARAN}:improved" if improved else component
        if chosen is None or chosen == "auto":
            wanted = phonemize.candidates(profile, language)
            installed = [c for c in wanted if self.is_installed(c)]
            chosen = installed[0] if installed else (wanted[0] if wanted else None)
        for option in options:
            option["default"] = option["id"] == chosen and (
                option["available"] or not any(o["available"] for o in options)
            )
        return options

    def requirements(self, spec: ModelSpec) -> tuple[list[str], list[dict[str, Any]], bool]:
        """``(requires, phonemizers, available)`` of a catalog entry, for this installation.

        ``requires`` names the components the model needs here and now: for a voice that reads
        phonemes, the phonemiser it would use (``uccharan`` or ``espeak-ng``); ``phonemizers`` lists
        the choices (see :meth:`phonemizer_options`); ``available`` is true when what is needed is installed.
        """
        requires: list[str] = []
        options: list[dict[str, Any]] = []
        available = True
        if spec.needs_phonemes:
            profile, language = spec.phoneme_target()
            options = self.phonemizer_options(profile, language, self.pinned_phonemizer(spec))
            default = next((o for o in options if o["default"]), None)
            component = (default["id"] if default else ESPEAK).split(":")[0]
            requires.append(component)
            available = bool(default and default["available"])
        for item in self._component_requires(spec):
            requires.append(item)
            available = available and self.is_installed(item)
        if spec.plugin:
            plugin = PLUGIN_PREFIX + spec.plugin
            requires.append(plugin)
            available = available and self.is_installed(plugin)
        return requires, options, available

    def _component_requires(self, spec: ModelSpec) -> list[str]:
        """Components named in ``requires`` that this installation needs (a server runtime is not needed when
        the operator points ``tts_servers`` at a running server for the model)."""
        out: list[str] = []
        for item in spec.requires:
            if item in ("phonemizer", ESPEAK):
                continue
            if item in tts_runtimes.RUNTIMES and tts_runtimes.remote_url(self.core.settings, spec):
                continue
            out.append(item)
        return out

    def voice_phonemizers(self, spec: ModelSpec) -> dict[str, list[dict[str, Any]]]:
        """Per-voice choices for a model whose voices differ in language (Kokoro)."""
        if spec.needs_phonemes or spec.kind != "kokoro":
            return {}
        pinned = self.pinned_phonemizer(spec)
        out: dict[str, list[dict[str, Any]]] = {}
        for v in spec.voices:
            target = self.phoneme_target(spec, str(v["id"]))
            if target:
                out[str(v["id"])] = self.phonemizer_options(*target, pinned=pinned)
        return out

    def check_phonemizer(self, spec: ModelSpec, voice_id: str | None, choice: str | None) -> str | None:
        """Validate a requested (or pinned) phonemiser for a voice; returns the choice to pass to the engine."""
        effective = choice or self.pinned_phonemizer(spec)
        if not effective:
            return None
        component, _ = phonemize.parse_choice(effective)  # 400 on a value that is not a phonemiser
        target = self.phoneme_target(spec, voice_id)
        if target is None or component == "auto":
            return None  # the voice reads characters or a lexicon: nothing to choose
        phonemize.resolve(*target, effective)  # 424 when not installed, 400 when uccharan lacks the language
        return effective

    def validate_pin(self, spec: ModelSpec, value: str) -> None:
        """A phonemiser can be pinned on a model whose voices read phonemes; uccharan only where it covers them."""
        from bolo.errors import InvalidRequest

        component, _ = phonemize.parse_choice(value)
        candidates = [self.phoneme_target(spec)] if spec.needs_phonemes else [
            self.phoneme_target(spec, str(v["id"])) for v in spec.voices
        ]  # fmt: skip
        targets = [t for t in candidates if t is not None]
        if not targets:
            raise InvalidRequest(f"model '{spec.id}' does not use a phonemiser", param="phonemizer")
        if component == UCCHARAN and not any(phonemize.uccharan_language(*t) for t in targets):
            raise InvalidRequest(
                f"uccharan does not cover the language of '{spec.id}' ({targets[0][1]})", param="phonemizer"
            )

    def missing_reason(self, spec: ModelSpec, choice: str | None = None) -> str | None:
        """Why a model cannot be used for lack of a component, or None.

        With a phonemiser chosen (by the request, else pinned on the model), that choice is checked instead,
        raising ``InvalidRequest`` / ``ComponentMissing`` itself.
        """
        if spec.plugin and not self.is_installed(PLUGIN_PREFIX + spec.plugin):
            hint = spec.options.get("plugin_install") or "see docs/plugins.md"
            return f"it runs on the engine plugin '{spec.plugin}', which is not installed ({hint})"
        if spec.needs_phonemes:
            profile, language = spec.phoneme_target()
            effective = choice or self.pinned_phonemizer(spec)
            if effective and phonemize.parse_choice(effective)[0] != "auto":
                phonemize.resolve(profile, language, effective)
            elif phonemize.select(profile, language) is None:
                return phonemize.missing_hint(profile, language)
        for item in self._component_requires(spec):
            if item in duplex_runtime.COMPONENTS:  # the build the model runs on, not any build
                if duplex_runtime.missing_reason(self.core.settings, spec) is None:
                    continue
            elif self.is_installed(item):
                continue
            if item in duplex_runtime.COMPONENTS:
                return duplex_runtime.missing_reason(self.core.settings, spec) or f"it needs the component '{item}'"
            if item in tts_runtimes.RUNTIMES:
                info = tts_runtimes.RUNTIME_INFO[item]
                return (
                    f"it runs on {info.name}, which is not installed. Install it with `bolo components install "
                    f"{item}` (or POST /api/v1/components/{item}/install), set {info.setting} to a binary, or point "
                    f'the setting tts_servers at a running server ("{spec.id}=http://host:port")'
                )
            return f"it needs the component '{item}', which is not installed"
        return None

    # -- policy -----------------------------------------------------------
    def install_blocked_reason(self) -> str | None:
        s = self.core.settings
        allowed = s.allow_component_install
        exposed = s.host not in LOOPBACK and not s.api_key
        if exposed:
            return "installing components is never allowed on a non-loopback address without an API key"
        if allowed is False:
            return "the operator disabled component installation (allow_component_install = false)"
        if installer_command() is None:
            return "neither pip nor uv is available in this environment"
        if not os.access(sysconfig.get_paths()["purelib"], os.W_OK):
            return "the Python environment is read-only"
        return None

    # -- status -----------------------------------------------------------
    def required_by(self, component: str) -> int:
        count = 0
        for spec in self.core.catalog.all():
            if spec.status == "pending":
                continue
            if component.startswith(PLUGIN_PREFIX):
                count += spec.engine == component
            elif spec.needs_phonemes:
                count += component in phonemize.candidates(*spec.phoneme_target())
            else:
                count += component in spec.requires
        return count

    def is_installed(self, component: str) -> bool:
        if component.startswith(XLIT_PREFIX):
            return romanised.pack_present(component[len(XLIT_PREFIX) :])
        name = packs.pack_of(component)
        if name is not None:
            return packs.installed(name)
        if component == ESPEAK:
            return espeak_status()[0]
        if component == UCCHARAN:
            return uccharan_status()[0]
        if component.startswith(PLUGIN_PREFIX):
            return self.core.engines.status(component).available
        if component in tts_runtimes.RUNTIMES:
            return tts_runtimes.find_binary(self.core.settings, component)[0] is not None
        if component in duplex_runtime.COMPONENTS:
            return bool(duplex_runtime.installed_flavours(self.core.settings, component))
        return False

    def plugin_ids(self) -> list[str]:
        ids = {e.id for e in self.core.engines.plugins()}
        ids |= {spec.engine for spec in self.core.catalog.all() if spec.plugin and spec.status != "pending"}
        return sorted(ids)

    def status(self) -> list[dict[str, Any]]:
        blocked = self.install_blocked_reason()
        out: list[dict[str, Any]] = []
        installed, source, version = uccharan_status()
        out.append(
            {
                "id": UCCHARAN,
                "name": "uccharan",
                "kind": "phonemizer",
                "installed": installed,
                "source": source,
                "version": version,
                "license": "Apache-2.0",
                "license_url": "https://github.com/cb-Sigma/uccharan/blob/main/LICENSE",
                "required_by_count": self.required_by(UCCHARAN),
                "installable": blocked is None and not installed,
                "install_blocked_reason": None if installed else blocked,
                "manual_command": manual_command(UCCHARAN),
                "languages": sorted({lang for table in phonemize.UCCHARAN_SUPPORT.values() for lang in table.values()}),
                "description": (
                    "Text-to-phoneme front end and text normaliser (Apache-2.0) for Piper-format, KittenTTS, "
                    "Matcha-TTS and Kokoro voices in 39 languages: English (US, British), Hindi and the other Indian "
                    "languages, Urdu, Sindhi, and Spanish, German, Dutch, French (also Belgian and Swiss), Italian, "
                    "Portuguese (Brazil, Portugal), Polish, Turkish, Czech, Swedish, Catalan, Romanian, Hungarian, "
                    "Finnish, Norwegian Bokmål and Danish; for Kokoro English a port of misaki, the front end Kokoro "
                    "was trained with. Languages other than American English and Hindi need a language pack, "
                    "downloaded on first use. Part of bolo-speech[all]. Voices in other languages need eSpeak NG."
                ),
            }
        )
        installed, source, version = espeak_status()
        out.append(
            {
                "id": ESPEAK,
                "name": "eSpeak NG",
                "kind": "phonemizer",
                "installed": installed,
                "source": source,
                "version": version,
                "license": "GPL-3.0-or-later",
                "license_url": "https://github.com/espeak-ng/espeak-ng/blob/master/COPYING",
                "required_by_count": self.required_by(ESPEAK),
                "installable": blocked is None and not installed,
                "install_blocked_reason": None if installed else blocked,
                "manual_command": manual_command(ESPEAK),
                "description": (
                    "Text-to-phoneme front end used by Piper-format voices, KittenTTS, Matcha-TTS and Kokoro in the "
                    "languages uccharan does not cover. Optional and not shipped with bolo: it is licensed "
                    "GPL-3.0-or-later and is installed only on request. Character- and lexicon-based voices "
                    "(MMS-TTS, SYSPIN, Kokoro Mandarin, MeloTTS) work without it."
                ),
            }
        )
        out.extend(self.pack_status())
        out.extend(self.runtime_status())
        out.extend(self.duplex_status())
        out.extend(self.torch_status())
        for plugin in self.plugin_ids():
            st = self.core.engines.status(plugin)
            hint = next(
                (str(s.options["plugin_install"]) for s in self.core.catalog.all()
                 if s.engine == plugin and s.options.get("plugin_install")),
                "see docs/plugins.md",
            )  # fmt: skip
            out.append(
                {
                    "id": plugin,
                    "name": f"engine plugin '{plugin[len(PLUGIN_PREFIX) :]}'",
                    "kind": "engine-plugin",
                    "installed": st.available,
                    "source": "plugin",
                    "version": st.version,
                    "license": "see the plugin package",
                    "license_url": None,
                    "required_by_count": self.required_by(plugin),
                    "installable": False,
                    "install_blocked_reason": None
                    if st.available
                    else "engine plugins are installed as packages by the operator",
                    "manual_command": hint,
                    "description": (
                        "An engine that runs in its own process (for example models that need PyTorch, which bolo "
                        "itself never loads). " + ("" if st.available else (st.reason or ""))
                    ).strip(),
                }
            )
        return out

    # -- uccharan's romanised-text packs ---------------------------------------
    def pack_blocked_reason(self) -> str | None:
        """Packs go into uccharan's cache directory, not into the Python environment."""
        s = self.core.settings
        if s.host not in LOOPBACK and not s.api_key:
            return "installing components is never allowed on a non-loopback address without an API key"
        if s.allow_component_install is False:
            return "the operator disabled component installation (allow_component_install = false)"
        return None

    def pack_status(self) -> list[dict[str, Any]]:
        """One component per pack the installed uccharan lists: its language packs (bolo.text.packs) and, when it
        has a transliterator, its romanised-text packs (bolo.text.romanised)."""
        blocked = self.pack_blocked_reason()
        out = self.language_pack_status(blocked)
        version = romanised.api().version
        sizes = {name: size for name, (_sha, size) in packs.table().items()}
        for language in romanised.pack_languages():
            installed = romanised.pack_present(language)
            out.append(
                {
                    "id": romanised.component_id(language),
                    "name": f"uccharan romanised-text pack ({language})",
                    "kind": "language-pack",
                    "installed": installed,
                    "source": "uccharan" if installed else None,
                    "version": version if installed else None,
                    "license": romanised.PACK_LICENSE,
                    "license_url": "https://github.com/cb-Sigma/uccharan/blob/main/DATA_LICENSES.md",
                    "built_in": False,
                    "required_by_count": 0,
                    "required_by_installed_count": 0,
                    "removable": installed
                    and blocked is None
                    and self.pack_remove_reason(romanised.pack_name(language)) is None,
                    "installable": blocked is None and not installed,
                    "install_blocked_reason": None if installed else blocked,
                    "manual_command": romanised.manual_command(language),
                    "size_bytes": sizes.get(romanised.pack_name(language)),
                    "auto_download": False,
                    "languages": [language],
                    "description": (
                        f"Data for the `romanised` option of text-to-speech: '{language}' text typed in Latin letters "
                        "is written in its own script before a voice reads it. Downloaded from uccharan's release, "
                        "checked against its SHA-256 and kept in uccharan's cache. Licensed CC BY-SA 4.0 (it is "
                        "built from share-alike data), so it is installed only on request."
                    ),
                }
            )
        return out

    def pack_users(self) -> tuple[dict[str, int], dict[str, int]]:
        """How many catalog models need each uccharan pack (through uccharan, as this installation phonemises), and
        how many of the installed ones do. A built-in pack (``en-us``, ``hi``) is never needed: it counts the models
        that read with its data instead."""
        counts: dict[str, int] = {}
        mine: dict[str, int] = {}
        installed = {str(m.get("id")) for m in self.core.store.installed()}
        for spec in self.core.catalog.all():
            if spec.status == "pending":
                continue
            targets = [spec.phoneme_target()] if spec.needs_phonemes else [
                self.phoneme_target(spec, str(v["id"])) for v in spec.voices
            ]  # fmt: skip
            names: set[str] = set()
            for target in targets:
                if target is None or UCCHARAN not in phonemize.candidates(*target):
                    continue
                tag = phonemize.uccharan_language(*target)
                if tag is None:
                    continue
                names.update(packs.static_needs(tag, target[0]))
                names.update(name for name, users in packs.BUILT_IN_USERS.items() if tag in users)
            for name in names:
                counts[name] = counts.get(name, 0) + 1
                if spec.id in installed:
                    mine[name] = mine.get(name, 0) + 1
        return counts, mine

    def language_pack_status(self, blocked: str | None) -> list[dict[str, Any]]:
        """uccharan's packs other than the romanised-text ones (those are listed by :meth:`pack_status`)."""
        version = packs.version()
        users, mine = self.pack_users() if packs.table() else ({}, {})
        out: list[dict[str, Any]] = []
        for pack in packs.all_packs():
            if pack.name.startswith(romanised.PACK_PREFIX):
                continue
            installed = packs.installed(pack.name)
            language = packs.LANGUAGE_NAMES.get(pack.language, pack.language)
            out.append(
                {
                    "id": pack.component,
                    "name": f"uccharan language pack '{pack.name}' ({language})",
                    "kind": "language-pack",
                    "installed": installed,
                    "source": "uccharan" if installed else None,
                    "version": version if installed else None,
                    "license": pack.license,
                    "license_url": packs.LICENSE_URL,
                    "size_bytes": pack.size,
                    "sha256": pack.sha256,
                    # a built-in pack is never needed (its data is part of uccharan): used_by_count counts the
                    # models that read with that data instead
                    "built_in": pack.name in packs.BUILT_IN,
                    "required_by_count": 0 if pack.name in packs.BUILT_IN else users.get(pack.name, 0),
                    "required_by_installed_count": 0 if pack.name in packs.BUILT_IN else mine.get(pack.name, 0),
                    "used_by_count": users.get(pack.name, 0) if pack.name in packs.BUILT_IN else None,
                    "removable": installed and blocked is None and self.pack_remove_reason(pack.name) is None,
                    "installable": blocked is None and not installed,
                    "install_blocked_reason": None if installed else blocked,
                    "manual_command": packs.manual_command(pack.name),
                    "auto_download": not pack.share_alike and pack.name not in packs.BUILT_IN and packs.auto_download(),
                    "languages": [pack.language],
                    "description": pack.description,
                }
            )
        return out

    # -- TTS server runtimes ------------------------------------------------
    def runtime_blocked_reason(self, runtime: str, flavour: str | None = None) -> str | None:
        s = self.core.settings
        unsupported = tts_runtimes.unsupported_reason(runtime)
        if unsupported:
            return unsupported
        if s.host not in LOOPBACK and not s.api_key:
            return "installing components is never allowed on a non-loopback address without an API key"
        if s.allow_component_install is False:
            return "the operator disabled component installation (allow_component_install = false)"
        info = tts_runtimes.RUNTIME_INFO[runtime]
        if info.git is not None:
            if shutil.which("git") is None:
                return "git is not installed (it fetches the pinned source)"
        elif flavour is not None and flavour not in tts_runtimes.pinned_flavours(runtime):
            pinned = ", ".join(tts_runtimes.pinned_flavours(runtime)) or "none"
            return f"no {flavour} build of {info.name} is pinned for {tts_runtimes.platform_key()} (pinned: {pinned})"
        elif tts_runtimes.release_for(runtime, s.device) is None:
            return (
                f"no prebuilt release is pinned for this platform ({tts_runtimes.platform_key()}); build "
                f"{info.binary} from {info.homepage} and set {info.setting} to its full path"
            )
        target = tts_runtimes.components_dir(s)
        probe = target if target.exists() else Path(s.data_dir)
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if not os.access(probe, os.W_OK):
            return f"the data directory is read-only ({probe})"
        return None

    def runtime_needs_install(self, runtime: str, flavour: str | None = None) -> bool:
        """Whether ``bolo components install <runtime>`` has something to do: the build ``flavour`` names (else the
        one ``device`` asks for) is not in the components folder, and no binary is configured in the settings. A
        binary on ``PATH`` does not count: the components folder comes before ``PATH``."""
        info = tts_runtimes.RUNTIME_INFO[runtime]
        if getattr(self.core.settings, info.setting, None):
            return False
        if info.git is not None:
            return tts_runtimes.find_binary(self.core.settings, runtime)[0] is None
        wanted = flavour or tts_runtimes.flavour_for(runtime, self.core.settings.device)
        return wanted not in tts_runtimes.installed_flavours(self.core.settings, runtime)

    def runtime_status(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for runtime in tts_runtimes.RUNTIMES:
            info = tts_runtimes.RUNTIME_INFO[runtime]
            state = tts_runtimes.install_status(self.core.settings, runtime)
            installed = state["path"] is not None
            blocked = self.runtime_blocked_reason(runtime)
            unsupported = tts_runtimes.unsupported_reason(runtime)
            release = state["release"]
            if info.git is not None:
                manual = (
                    f"bolo components install {runtime} (or git clone --branch {info.git.tag} {info.git.url} and set "
                    f"{info.setting}); uv must be on PATH"
                )
            elif release:
                manual = f"bolo components install {runtime}"
            else:
                manual = f"build {info.binary} from {info.homepage} and set {info.setting}"
            out.append(
                {
                    "id": runtime,
                    "name": info.name,
                    "kind": "tts-server",
                    "installed": installed,
                    "source": state["source"],
                    "version": info.version if state["source"] == "bolo" else ("found" if installed else None),
                    "license": info.license,
                    "license_url": info.license_url,
                    "required_by_count": self.required_by(runtime),
                    "installable": blocked is None and self.runtime_needs_install(runtime),
                    "install_blocked_reason": None if installed else blocked,
                    # False where the runtime cannot run at all (Tontaube off Linux), with the reason
                    "available": unsupported is None,
                    "unavailable_reason": unsupported,
                    # the build that runs (from bolo's components folder): cpu, cuda or vulkan
                    "flavour": state["flavour"],
                    "flavours": state["flavours"],
                    "wanted_flavour": state["wanted_flavour"] if info.git is None else None,
                    "manual_command": manual
                    + f'; or run the server yourself and set tts_servers = ["{runtime}=http://host:port"]',
                    "path": state["path"],
                    "servers": state["urls"],
                    "release": {"version": info.version, "url": release.url, "sha256": release.sha256}
                    if release
                    else (
                        {"version": info.version, "url": info.git.url, "commit": info.git.commit}
                        if info.git is not None
                        else None
                    ),
                    "description": info.description,
                }
            )
        return out

    # -- full-duplex model servers (bolo.duplex.runtime) ----------------------------------------------------------
    def duplex_blocked_reason(self, component: str, flavour: str | None = None) -> str | None:
        s = self.core.settings
        if s.host not in LOOPBACK and not s.api_key:
            return "installing components is never allowed on a non-loopback address without an API key"
        if s.allow_component_install is False:
            return "the operator disabled component installation (allow_component_install = false)"
        return duplex_runtime.blocked_reason(component, flavour)

    def duplex_status(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for component in duplex_runtime.COMPONENTS:
            info = duplex_runtime.SERVERS[component]
            state = duplex_runtime.status(self.core.settings, component)
            installed = bool(state["flavours"])
            blocked = self.duplex_blocked_reason(component)
            unsupported = duplex_runtime.unsupported_reason(component)
            wanted = state["wanted_flavour"]
            build = info.builds.get(wanted or "")
            out.append(
                {
                    "id": component,
                    "name": info.name,
                    "kind": "duplex-server",
                    "installed": installed,
                    "source": "bolo" if installed else None,
                    "version": ", ".join(f"{f} {info.builds[f].version}" for f in state["flavours"]) or None,
                    "license": info.license,
                    "license_url": info.license_url,
                    "required_by_count": self.required_by(component),
                    "installable": blocked is None and (wanted not in state["flavours"]),
                    "install_blocked_reason": None if installed else blocked,
                    "available": unsupported is None,
                    "unavailable_reason": unsupported,
                    "flavour": wanted if wanted in state["flavours"] else (state["flavours"] or [None])[0],
                    "flavours": state["flavours"],
                    "wanted_flavour": wanted,
                    "builds": state["builds"],
                    "manual_command": f"bolo components install {component}"
                    + (
                        ""
                        if not build
                        else f" (the {build.id} build; --flavour picks another: {', '.join(info.builds)})"
                    ),
                    "path": state["path"],
                    "servers": [self.core.settings.duplex_server_url] if self.core.settings.duplex_server_url else [],
                    "release": None,
                    "description": info.description,
                }
            )
        return out

    def install_runtime_blocking(self, runtime: str, job: Job | None = None, flavour: str | None = None) -> str | None:
        """Download the pinned release archive for this platform, check its SHA-256 and unpack it into the folder of
        its build (``flavour``, else the one ``device`` asks for: ``<runtime>/<version>/cpu|cuda|vulkan|rocm``)."""
        from bolo.store import RemoteFile

        info = tts_runtimes.RUNTIME_INFO[runtime]
        if info.git is not None:
            return self._install_checkout(runtime)
        if flavour is None:
            note = tts_runtimes.missing_build_note(runtime, self.core.settings.device)
            if note:
                log.warning("%s", note)
                if job is not None:
                    job.warnings.append({"kind": "build", "message": note})
        flavour = flavour or tts_runtimes.flavour_for(runtime, self.core.settings.device)
        release = tts_runtimes.release_for(runtime, flavour=flavour)
        if release is None:
            return f"no prebuilt {flavour} {info.name} release is pinned for {tts_runtimes.platform_key()}"
        tts_runtimes.migrate_legacy(self.core.settings, runtime)
        archives = [release, *release.extras]
        total = sum(a.size or 0 for a in archives) if all(a.size for a in archives) else 0

        def progress(n: int) -> None:
            if job is not None and total:
                job.bytes_done += n
                job.bytes_total = total
                job.progress = min(0.95, job.bytes_done / total)

        target = tts_runtimes.install_dir(self.core.settings, runtime, flavour)
        tmp = target.with_name(target.name + ".partial")
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            for archive in archives:
                name = archive.url.rsplit("/", 1)[-1]
                blob, _ = self.core.store.fetch_blob(
                    RemoteFile(name, archive.url, archive.size, archive.sha256), on_bytes=progress
                )
                tts_runtimes.extract_release(blob, tmp)
                blob.unlink(missing_ok=True)
            shutil.rmtree(target, ignore_errors=True)
            tmp.rename(target)
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            return f"{type(exc).__name__}: {exc}"
        path = tts_runtimes.installed_flavours(self.core.settings, runtime).get(flavour)
        if path is None:
            return f"the release was unpacked but {info.binary} was not found in it"
        problem = tts_runtimes.check_runs(path) if path.is_relative_to(target) else None
        if problem:
            shutil.rmtree(target, ignore_errors=True)
            return f"{info.binary} {info.version} does not run here: {problem}"
        return None

    def _install_checkout(self, runtime: str) -> str | None:
        """Clone a runtime's pinned tag (shallow) and check that it is the pinned commit."""
        info = tts_runtimes.RUNTIME_INFO[runtime]
        git = info.git
        assert git is not None
        target = tts_runtimes.install_dir(self.core.settings, runtime) / info.binary
        tmp = target.with_name(target.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        code, output = self._run(["git", "clone", "--quiet", "--depth", "1", "--branch", git.tag, git.url, str(tmp)])
        if code != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            return f"git clone failed ({code}): {' | '.join(output.strip().splitlines()[-3:])}"
        code, head = self._run(["git", "-C", str(tmp), "rev-parse", "HEAD"])
        if code != 0 or head.strip() != git.commit:
            shutil.rmtree(tmp, ignore_errors=True)
            return f"{git.url} {git.tag} is not the pinned commit {git.commit} (got {head.strip()[:40]})"
        shutil.rmtree(target, ignore_errors=True)
        tmp.rename(target)
        return None

    # -- remove -----------------------------------------------------------
    @staticmethod
    def pack_remove_reason(name: str) -> str | None:
        """Why a pack's file cannot be deleted now (None: it can), see ``packs.removable``."""
        if name.startswith(romanised.PACK_PREFIX) and name in romanised._loaded:
            return "this server has loaded it; restart the server first"
        return packs.removable(name)

    def remove_pack(self, component: str) -> None:
        """Delete a downloaded uccharan pack from uccharan's cache; the only components removed from the API."""
        name = packs.pack_of(component)
        if name is None:
            if (
                component in PACKAGES
                or component in tts_runtimes.RUNTIMES
                or component in duplex_runtime.COMPONENTS
                or component.startswith(PLUGIN_PREFIX)
            ):
                raise InvalidRequest(f"only uccharan's language packs can be removed from the API ({component})")
            raise ModelNotFound(f"unknown component '{component}'")
        blocked = self.pack_blocked_reason()
        reason = blocked or (None if self.is_installed(component) else "it is not installed")
        reason = reason or self.pack_remove_reason(name)
        if reason is not None:
            raise InvalidRequest(f"cannot remove {component}: {reason}")
        try:
            packs.remove(name)
        except (ValueError, OSError) as exc:
            raise InvalidRequest(f"cannot remove {component}: {exc}") from exc
        self.core.events.publish("component", {"id": component, "event": "removed"})

    # -- GPU builds of PyTorch (bolo.torch_envs) ----------------------------------------------------------
    def torch_state(self, fresh: bool = False) -> tuple[list[torch_envs.TorchEnv], torch_envs.Gpu]:
        """The bolo-torch environments and the GPU, looked at again at most every TORCH_STATUS_TTL_S."""
        import time

        cached = self._torch_cache
        if cached is not None and not fresh and time.monotonic() - cached[0] < TORCH_STATUS_TTL_S:
            return cached[1], cached[2]
        s = self.core.settings
        commands = torch_envs.plugin_commands(s.plugins, s.plugin_entry_points)
        envs = torch_envs.discover(commands, self.torch_runner)
        gpu = torch_envs.detect_gpu(self.torch_runner)
        self._torch_cache = (time.monotonic(), envs, gpu)
        return envs, gpu

    def torch_status(self) -> list[dict[str, Any]]:
        """``torch-<target>`` for the GPU this machine has, when a bolo-torch plugin is installed."""
        try:
            envs, gpu = self.torch_state()
        except Exception as exc:  # a status page must not fail on it
            log.debug("could not look at the bolo-torch environments: %s", exc)
            return []
        if not envs or gpu.target is None:
            return []
        target = gpu.target
        blocked = self.torch_blocked_reason()
        installed = all(torch_envs.family_of(e.build) == target for e in envs)
        label = {"cuda": "CUDA", "rocm": "ROCm", "xpu": "Intel XPU"}.get(target, target)
        lines = "; ".join(f"{', '.join(e.plugins)}: torch {e.torch or '-'}, {e.to_dict(gpu)['label']}" for e in envs)
        return [
            {
                "id": TORCH_PREFIX + target,
                "name": f"PyTorch, {label} build",
                "kind": "pytorch",
                "installed": installed,
                "source": "python-package" if installed else None,
                "version": ", ".join(sorted({e.torch for e in envs if e.torch})) or None,
                "license": "BSD-3-Clause",
                "license_url": "https://github.com/pytorch/pytorch/blob/main/LICENSE",
                "required_by_count": sum(self.required_by(p) for p in {p for e in envs for p in e.plugins}),
                "installable": blocked is None and not installed,
                "install_blocked_reason": None if installed else blocked,
                "manual_command": f"bolo torch use {target}",
                "size_bytes": torch_envs.DOWNLOAD_BYTES,
                "gpu": gpu.name or gpu.vendor,
                "envs": [e.to_dict(gpu) for e in envs],
                "description": (
                    f"The {label} build of PyTorch for the bolo-torch plugin, so its models use the "
                    f"{gpu.name or gpu.vendor or 'GPU'}. PyPI's torch is CPU only on Windows and macOS. Installing "
                    "reinstalls torch (and torchaudio and torchvision where present) from PyTorch's index in every "
                    f"bolo-torch environment, checks that it reaches the GPU and restarts the plugin. Now: {lines}."
                ),
            }
        ]

    def torch_blocked_reason(self) -> str | None:
        s = self.core.settings
        if s.host not in LOOPBACK and not s.api_key:
            return "installing components is never allowed on a non-loopback address without an API key"
        if s.allow_component_install is False:
            return "the operator disabled component installation (allow_component_install = false)"
        return None

    async def stop_plugins(self, plugins: list[str]) -> None:
        """Unload the models of ``plugins`` and stop their processes (started again by the next load)."""
        for model_id, entry in list(self.core.manager.loaded.items()):
            if entry.spec.engine in plugins:
                await self.core.manager.unload(model_id, reason="the plugin's torch is being replaced")
        for plugin in plugins:
            engine = self.core.engines.get(plugin)
            process = getattr(engine, "process", None)
            if process is not None:
                await asyncio.to_thread(process.stop)

    async def _install_torch(self, job: Job, target: str) -> None:
        core = self.core
        loop = asyncio.get_running_loop()
        job.state, job.progress = "running", 0.02
        core.events.publish("job", {"job": job.to_dict()})

        def stop(plugins: list[str]) -> None:
            asyncio.run_coroutine_threadsafe(self.stop_plugins(plugins), loop).result(timeout=300)

        def progress(fraction: float, what: str) -> None:
            job.progress = 0.05 + 0.9 * fraction
            core.events.publish("job", {"job": job.to_dict()})
            core.log("info", f"PyTorch {target} build: {what}")

        def work() -> list[dict[str, Any]]:
            envs, gpu = self.torch_state(fresh=True)
            plans = [torch_envs.plan(e, target, gpu, self.torch_page) for e in envs]
            return torch_envs.apply(plans, self.torch_runner, stop, progress)

        try:
            results = await loop.run_in_executor(None, work)
        except Exception as exc:
            results = [{"ok": False, "detail": f"{type(exc).__name__}: {exc}", "plugins": []}]
        self._torch_cache = None
        failed = [r for r in results if not r["ok"]]
        if not results:
            job.state, job.error = "error", "no environment with the bolo-torch plugin was found"
        elif failed:
            job.state = "error"
            job.error = "; ".join(f"{', '.join(r['plugins'])}: {r['detail']}" for r in failed)
            core.log("error", f"installing the PyTorch {target} build failed: {job.error}")
        else:
            job.state, job.progress = "done", 1.0
            core.events.publish("component", {"id": job.model, "event": "installed"})
            core.log("info", "PyTorch " + target + " build: " + "; ".join(str(r["detail"]) for r in results))
        core.events.publish("job", {"job": job.to_dict()})

    # -- install ----------------------------------------------------------
    def start_install(self, component: str, accept_license: bool, flavour: str | None = None) -> Job:
        """Start installing ``component``; ``flavour`` (``cpu``, ``cuda``, ``vulkan``, ``rocm``) picks the build of
        llama-server or audio.cpp, which is otherwise the one ``device`` asks for."""
        if component in TORCH_COMPONENTS:
            blocked = self.torch_blocked_reason()
            if blocked:
                raise InvalidRequest(f"cannot install {component}: {blocked}")
            job = self.core.jobs.create("install", component)
            task = asyncio.create_task(
                self._install_torch(job, component[len(TORCH_PREFIX) :]), name=f"bolo-install-{component}"
            )
            self.core._tasks.append(task)
            task.add_done_callback(lambda t: self.core._tasks.remove(t) if t in self.core._tasks else None)
            return job
        if component in duplex_runtime.COMPONENTS:
            blocked = self.duplex_blocked_reason(component, flavour)
            if blocked:
                raise InvalidRequest(f"cannot install {component}: {blocked}")
            job = self.core.jobs.create("install", component)
            build = duplex_runtime.build_of(component, flavour)
            if build is not None and build.id in duplex_runtime.installed_flavours(self.core.settings, component):
                job.state, job.progress = "done", 1.0
                return job
            task = asyncio.create_task(self._install(job, flavour), name=f"bolo-install-{component}")
            self.core._tasks.append(task)
            task.add_done_callback(lambda t: self.core._tasks.remove(t) if t in self.core._tasks else None)
            return job
        if flavour is not None and component not in tts_runtimes.RUNTIMES:
            raise InvalidRequest(
                f"{component} has no builds to choose from (flavour is for llama-server, audio.cpp, moshi, personaplex)"
            )
        if flavour is not None and flavour not in tts_runtimes.FLAVOURS:
            raise InvalidRequest(f"unknown flavour {flavour!r}: one of {', '.join(tts_runtimes.FLAVOURS)}")
        if component in tts_runtimes.RUNTIMES:
            blocked = self.runtime_blocked_reason(component, flavour)
            if blocked:
                raise InvalidRequest(f"cannot install {component}: {blocked}")
            job = self.core.jobs.create("install", component)
            if not self.runtime_needs_install(component, flavour):
                job.state, job.progress = "done", 1.0
                return job
            task = asyncio.create_task(self._install(job, flavour), name=f"bolo-install-{component}")
            self.core._tasks.append(task)
            task.add_done_callback(lambda t: self.core._tasks.remove(t) if t in self.core._tasks else None)
            return job
        if component.startswith(XLIT_PREFIX):
            return self._start_pack_install(component, accept_license)
        if packs.pack_of(component) is not None:
            return self._start_language_pack_install(component, accept_license)
        if component not in PACKAGES:
            if component.startswith(PLUGIN_PREFIX):
                raise InvalidRequest(f"engine plugins are installed as packages, not from the API ({component})")
            raise ModelNotFound(f"unknown component '{component}'")
        if component == ESPEAK and accept_license is not True:
            raise InvalidRequest(
                'eSpeak NG is licensed GPL-3.0-or-later; send {"accept_license": true} to install it',
                param="accept_license",
            )
        blocked = self.install_blocked_reason()
        if blocked:
            raise InvalidRequest(f"cannot install {component}: {blocked}. Manual route: {manual_command(component)}")
        job = self.core.jobs.create("install", component)
        if self.is_installed(component):
            job.state, job.progress = "done", 1.0
            return job
        task = asyncio.create_task(self._install(job), name=f"bolo-install-{component}")
        self.core._tasks.append(task)
        task.add_done_callback(lambda t: self.core._tasks.remove(t) if t in self.core._tasks else None)
        return job

    def _start_pack_install(self, component: str, accept_license: bool) -> Job:
        language = component[len(XLIT_PREFIX) :]
        if language not in romanised.pack_languages():
            reason = romanised.api().reason or f"the installed uccharan has no pack for '{language}'"
            raise ModelNotFound(f"unknown component '{component}': {reason}")
        if accept_license is not True:
            raise InvalidRequest(
                f"the romanised-text packs are licensed {romanised.PACK_LICENSE}; "
                'send {"accept_license": true} to install one',
                param="accept_license",
            )
        blocked = self.pack_blocked_reason()
        if blocked:
            raise InvalidRequest(
                f"cannot install {component}: {blocked}. Manual route: {romanised.manual_command(language)}"
            )
        job = self.core.jobs.create("install", component)
        if self.is_installed(component):
            job.state, job.progress = "done", 1.0
            return job
        task = asyncio.create_task(self._install(job), name=f"bolo-install-{component}")
        self.core._tasks.append(task)
        task.add_done_callback(lambda t: self.core._tasks.remove(t) if t in self.core._tasks else None)
        return job

    def _start_language_pack_install(self, component: str, accept_license: bool) -> Job:
        name = packs.pack_of(component)
        assert name is not None
        if packs.is_share_alike(name) and accept_license is not True:
            raise InvalidRequest(
                f"uccharan's pack '{name}' is licensed {packs.SHARE_ALIKE}; send {{\"accept_license\": true}} to "
                "install it",
                param="accept_license",
            )
        blocked = self.pack_blocked_reason()
        if blocked:
            raise InvalidRequest(f"cannot install {component}: {blocked}. Manual route: {packs.manual_command(name)}")
        job = self.core.jobs.create("install", component)
        if self.is_installed(component):
            job.state, job.progress = "done", 1.0
            return job
        task = asyncio.create_task(self._install(job), name=f"bolo-install-{component}")
        self.core._tasks.append(task)
        task.add_done_callback(lambda t: self.core._tasks.remove(t) if t in self.core._tasks else None)
        return job

    def install_pack_blocking(self, name: str, job: Job | None = None) -> str | None:
        """Download one uccharan pack (checked against its SHA-256) and load it. Returns an error, or None."""
        pack = packs.info(name)
        if pack is None:
            return f"the installed uccharan has no pack '{name}'"

        def progress(n: int) -> None:
            if job is not None:
                job.bytes_done += n
                job.bytes_total = pack.size
                job.progress = min(0.95, job.bytes_done / max(1, pack.size))

        try:
            packs.install(name, progress)
        except BoloError as exc:
            return exc.message
        if name.startswith(romanised.PACK_PREFIX):
            romanised.refresh()
        return None

    def install_blocking(
        self, job: Job | None = None, component: str = ESPEAK, flavour: str | None = None
    ) -> str | None:
        """Install the component's package and activate it. Returns an error message, or None on success."""
        if component in tts_runtimes.RUNTIMES:
            return self.install_runtime_blocking(component, job, flavour)
        if component in duplex_runtime.COMPONENTS:
            return duplex_runtime.install(
                self.core.settings, component, flavour, duplex_runtime.default_fetch(self.core.store),
                self.run_duplex_step, job,
            )  # fmt: skip
        if component.startswith(XLIT_PREFIX):
            return romanised.install_pack(component[len(XLIT_PREFIX) :])
        name = packs.pack_of(component)
        if name is not None:
            return self.install_pack_blocking(name, job)
        cmd = installer_command()
        if cmd is None:
            return "neither pip nor uv is available in this environment"
        code, output = self.run_installer([*cmd, PACKAGES[component]])
        if code != 0:
            tail = " | ".join(output.strip().splitlines()[-4:])
            return f"the installer exited with {code}: {tail}"
        importlib.invalidate_caches()
        phonemize.reset_phonemizer()
        packs.reset()
        romanised.refresh()  # a newer uccharan may bring the transliterator
        if not self.is_installed(component):
            return f"the package was installed but {component} could not be loaded"
        return None

    async def _install(self, job: Job, flavour: str | None = None) -> None:
        core = self.core
        job.state, job.progress = "running", 0.1
        core.events.publish("job", {"job": job.to_dict()})
        error = await asyncio.get_running_loop().run_in_executor(None, self.install_blocking, job, job.model, flavour)
        if error:
            job.state, job.error = "error", error
            core.log("error", f"installing {job.model} failed: {error}")
        else:
            job.state, job.progress = "done", 1.0
            with contextlib.suppress(Exception):  # the hardware report probes the new runtime afresh
                core.advisor.components_changed()
            core.events.publish("component", {"id": job.model, "event": "installed"})
            core.log("info", f"installed component {job.model}; voices that need it are now available")
        core.events.publish("job", {"job": job.to_dict()})
