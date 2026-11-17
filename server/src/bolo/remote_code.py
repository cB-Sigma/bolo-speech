"""Hugging Face "remote code": Python files in a model repository that the model's library imports and runs.

Some models exist only as code in their own Hugging Face repository (``trust_remote_code=True`` in transformers):
loading them runs that code with the full rights of the process that loads it. bolo never does that by default.
A catalog entry that needs it says so in ``options.remote_code``::

    remote_code:
      revision: <the commit of the repository whose code was reviewed>   # must be source.revision
      reviewed: '2026-11-17: what was read, and what the code does besides modelling (downloads, files, ...)'
      files:                                                              # sha256 of every .py file, and of every
        modeling_x.py: <sha256>                                           # JSON file that maps classes (auto_map)
        config.json: <sha256>

and the operator opts in for that one model with the per-model setting ``allow_remote_code`` (``bolo models config
<id> --set allow_remote_code=true`` or ``PUT /api/v1/models/<id>/settings``). The plugin that loads the model checks
the pin again before anything is imported: the revision, the sha256 of every listed file, no unlisted ``.py`` file and
no ``auto_map`` pointing at another repository (bolo-torch's ``remote_code`` module). The opt-in is the operator's;
a catalog entry cannot give it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec

SETTING = "allow_remote_code"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

EXPLAINED = (
    "Remote code means Python files from the model's own repository, imported and run by the plugin with the "
    "rights of its process (files, network, everything the account running bolo can do). bolo runs it only after "
    "you opt in for this model, and only the files reviewed at the pinned commit (each checked by sha256 before it "
    "is imported)."
)


def pin(spec: ModelSpec) -> dict[str, Any] | None:
    """The entry's remote-code pin (``options.remote_code``), or None for a model without remote code."""
    raw = spec.options.get("remote_code")
    return raw if isinstance(raw, dict) else None


def check_pin(spec: ModelSpec) -> str | None:
    """What is wrong with an entry's remote-code pin, or None (catalog validation)."""
    raw = spec.options.get("remote_code")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return "options.remote_code must be a mapping"
    revision = str(raw.get("revision") or "")
    if not _COMMIT.match(revision):
        return "options.remote_code.revision must be a 40-character commit"
    if spec.source.repo and spec.source.revision != revision:
        return f"options.remote_code.revision {revision} is not the source revision {spec.source.revision}"
    files = raw.get("files")
    if not isinstance(files, dict) or not files:
        return "options.remote_code.files must map each code file to its sha256"
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str) or not _SHA256.match(digest):
            return f"options.remote_code.files: {name!r} needs a lowercase sha256"
    if not any(str(n).endswith(".py") for n in files):
        return "options.remote_code.files lists no .py file"
    return None


def allowed(settings: dict[str, Any]) -> bool:
    """Whether the operator opted in (the per-model setting, never the catalog)."""
    return settings.get(SETTING) is True


def notice(spec: ModelSpec, settings: dict[str, Any] | None = None) -> str | None:
    """The warning shown on a pull or a load of a model with remote code; None for other models."""
    p = pin(spec)
    if p is None:
        return None
    files = ", ".join(sorted(str(n) for n in p.get("files", {}) if str(n).endswith(".py")))
    head = (
        f"{spec.id} runs remote code: {files} from {spec.source.repo or 'its repository'} at commit "
        f"{str(p.get('revision', ''))[:12]}. {EXPLAINED}"
    )
    if settings is not None and allowed(settings):
        return head + " You opted in for this model (allow_remote_code = true)."
    return (
        head + f" It will not load until you opt in: bolo models config {spec.id} --set allow_remote_code=true "
        "(read the files first; see docs/models.md, 'Remote code')."
    )
