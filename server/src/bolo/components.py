"""Optional third-party components that bolo can use but does not ship.

There is one today: **eSpeak NG**, the phonemiser that Piper-format voices,
KittenTTS, Matcha-TTS (English) and the non-English Kokoro voices were
trained with. It is GPL-3.0-or-later. bolo's wheel and its default extras do
not contain it; it is installed only when the operator asks for it, through
the console, ``bolo components install espeak-ng``, the ``espeak`` extra, or
the system package manager.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bolo.errors import InvalidRequest, ModelNotFound
from bolo.jobs import Job
from bolo.text import phonemize

if TYPE_CHECKING:
    from bolo.core import Bolo

ESPEAK = "espeak-ng"
ESPEAK_PACKAGE = "espeakng-loader>=0.2.4"
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def espeak_status() -> tuple[bool, str | None, str | None]:
    """``(installed, source, version)`` of the espeak-ng the phonemiser would use."""
    ph = phonemize.get_phonemizer()
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


def manual_command() -> str:
    """How a person would install the component for the way bolo itself was installed."""
    prefix = sys.prefix.replace("\\", "/")
    if Path("/.dockerenv").exists():
        return "rebuild the image with: docker build --build-arg WITH_ESPEAK=1 ..."
    if "/pipx/venvs/" in prefix:
        return "pipx inject bolo espeakng-loader"
    if "/uv/tools/" in prefix:
        return "uv tool install --with espeakng-loader bolo"
    return "pip install 'bolo[espeak]'"


class ComponentManager:
    def __init__(self, core: Bolo) -> None:
        self.core = core
        #: replaced in tests; returns (exit code, output)
        self.run_installer: Callable[[list[str]], tuple[int, str]] = self._run

    @staticmethod
    def _run(cmd: list[str]) -> tuple[int, str]:
        proc = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        return proc.returncode, (proc.stdout + proc.stderr).decode(errors="replace")

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
        return sum(1 for spec in self.core.catalog.all() if component in spec.requires)

    def is_installed(self, component: str) -> bool:
        return component == ESPEAK and espeak_status()[0]

    def status(self) -> list[dict[str, Any]]:
        installed, source, version = espeak_status()
        blocked = self.install_blocked_reason()
        return [
            {
                "id": ESPEAK,
                "name": "eSpeak NG",
                "installed": installed,
                "source": source,
                "version": version,
                "license": "GPL-3.0-or-later",
                "license_url": "https://github.com/espeak-ng/espeak-ng/blob/master/COPYING",
                "required_by_count": self.required_by(ESPEAK),
                "installable": blocked is None and not installed,
                "install_blocked_reason": None if installed else blocked,
                "manual_command": manual_command(),
                "description": (
                    "Text-to-phoneme front end used by Piper-format voices, KittenTTS, Matcha-TTS (English) and "
                    "the non-English Kokoro voices. Optional and not shipped with bolo: it is licensed "
                    "GPL-3.0-or-later and is installed only on request. Kokoro English and Mandarin, MeloTTS, "
                    "MMS-TTS and the Coqui voices work without it."
                ),
            }
        ]

    # -- install ----------------------------------------------------------
    def start_install(self, component: str, accept_license: bool) -> Job:
        if component != ESPEAK:
            raise ModelNotFound(f"unknown component '{component}'")
        if accept_license is not True:
            raise InvalidRequest(
                'eSpeak NG is licensed GPL-3.0-or-later; send {"accept_license": true} to install it',
                param="accept_license",
            )
        blocked = self.install_blocked_reason()
        if blocked:
            raise InvalidRequest(f"cannot install {component}: {blocked}. Manual route: {manual_command()}")
        job = self.core.jobs.create("install", component)
        if self.is_installed(component):
            job.state, job.progress = "done", 1.0
            return job
        task = asyncio.create_task(self._install(job), name=f"bolo-install-{component}")
        self.core._tasks.append(task)
        task.add_done_callback(lambda t: self.core._tasks.remove(t) if t in self.core._tasks else None)
        return job

    def install_blocking(self, job: Job | None = None) -> str | None:
        """Install the loader package and activate it. Returns an error message, or None on success."""
        cmd = installer_command()
        if cmd is None:
            return "neither pip nor uv is available in this environment"
        code, output = self.run_installer([*cmd, ESPEAK_PACKAGE])
        if code != 0:
            tail = " | ".join(output.strip().splitlines()[-4:])
            return f"the installer exited with {code}: {tail}"
        importlib.invalidate_caches()
        phonemize.reset_phonemizer()
        if phonemize.get_phonemizer() is None:
            return "the package was installed but the espeak-ng library could not be loaded"
        return None

    async def _install(self, job: Job) -> None:
        core = self.core
        job.state, job.progress = "running", 0.1
        core.events.publish("job", {"job": job.to_dict()})
        error = await asyncio.get_running_loop().run_in_executor(None, self.install_blocking, job)
        if error:
            job.state, job.error = "error", error
            core.log("error", f"installing {job.model} failed: {error}")
        else:
            job.state, job.progress = "done", 1.0
            core.events.publish("component", {"id": job.model, "event": "installed"})
            core.log(
                "info", f"installed component {job.model} (GPL-3.0-or-later); voices that need it are now available"
            )
        core.events.publish("job", {"job": job.to_dict()})
