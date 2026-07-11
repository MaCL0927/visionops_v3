"""Storage helpers for server-side dataset materialization.

Images are immutable after upload/review, so a hard link can expose the same
bytes in both ``batches`` and ``datasets`` without allocating a second copy.
When hard links are unavailable (for example across filesystems), the helper
falls back to a normal copy so existing behaviour is preserved.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


def link_or_copy_immutable(src: Path, dst: Path) -> dict[str, Any]:
    """Create *dst* from immutable *src*, preferring a hard link.

    Returns a small report with ``mode`` (``hardlink`` or ``copy``) and file
    size.  The destination is always replaced atomically enough for the
    dataset build path: callers create a fresh dataset directory first.
    """

    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"源文件不存在: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    size = int(src.stat().st_size)
    try:
        # resolve() makes a hard link to the actual image rather than to a
        # symlink inode that may live outside the managed data root.
        os.link(str(src.resolve()), str(dst))
        return {"mode": "hardlink", "size_bytes": size}
    except OSError as error:
        shutil.copy2(src, dst)
        return {
            "mode": "copy",
            "size_bytes": size,
            "fallback_error": f"{type(error).__name__}: {error}",
        }


def same_inode(path_a: Path, path_b: Path) -> bool:
    """Return True when both paths point at the same filesystem inode."""

    try:
        stat_a = Path(path_a).stat()
        stat_b = Path(path_b).stat()
    except OSError:
        return False
    return stat_a.st_dev == stat_b.st_dev and stat_a.st_ino == stat_b.st_ino
