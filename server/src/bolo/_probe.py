"""Load a model in a throw-away process: ``python -m bolo._probe <data_dir> <model_id>``.

Native inference libraries sometimes abort the whole process when a model
file is not what they expect (a wrong export, a truncated download, missing
metadata). The server therefore loads every newly installed model once in a
child process first. If the child dies, the server reports an error for
that model instead of going down with it.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write("usage: python -m bolo._probe <data_dir> <model_id> <threads>\n")
        return 2
    from bolo.catalog import Catalog
    from bolo.engines import EngineRegistry
    from bolo.engines.base import LoadOptions
    from bolo.store import ModelStore

    data_dir, model_id, threads = Path(argv[0]), argv[1], int(argv[2])
    spec = Catalog.load(data_dir).resolve(model_id)
    if spec is None:
        sys.stderr.write(f"unknown model {model_id}\n")
        return 3
    engine = EngineRegistry().get(spec.engine)
    if engine is None:
        sys.stderr.write(f"unknown engine {spec.engine}\n")
        return 3
    options = dict(spec.options)
    if spec.voices:
        options.setdefault("voices", spec.voices)
    model = engine.load(
        spec.kind,
        spec.task,
        ModelStore(data_dir).model_dir(spec.preset_of or spec.id),
        options,
        LoadOptions(threads=threads),
    )
    model.close()
    sys.stdout.write("ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
