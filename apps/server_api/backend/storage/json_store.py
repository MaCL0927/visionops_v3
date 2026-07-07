"""轻量 JSON 持久化工具。"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


class JsonStore:
    """基于单个 JSON 文件的线程内安全读写。"""

    def __init__(self, path: Path, *, default: Any) -> None:
        self.path = Path(path)
        self.default = default
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> Any:
        with self._lock:
            if not self.path.exists():
                return _clone(self.default)
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return _clone(self.default)

    def write(self, value: Any) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(tmp_name, self.path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)

    def update(self, mutator):  # type: ignore[no-untyped-def]
        with self._lock:
            value = self.read()
            result = mutator(value)
            self.write(value)
            return result


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))
