from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_verify_rgbd_cache_direct_entrypoint_imports_project_package() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "production/foam_ring_grasp/scripts/verify_rgbd_cache.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert "M36.3" in completed.stdout
