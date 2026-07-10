"""Shared C++ build fixtures for integration tests.

The Runtime used to be configured and compiled independently in every test
module.  A full test run therefore rebuilt the same target seven times.  Keep
one session build directory and share its binaries across all modules.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def shared_runtime_build_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build_dir = tmp_path_factory.mktemp("runtime-shared-build")
    subprocess.run(
        ["cmake", "-S", str(PROJECT_ROOT), "-B", str(build_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return build_dir


@pytest.fixture(scope="session")
def shared_runtime_binary(shared_runtime_build_dir: Path) -> Path:
    subprocess.run(
        [
            "cmake",
            "--build",
            str(shared_runtime_build_dir),
            "-j4",
            "--target",
            "visionops_runtime_mock",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    binary = shared_runtime_build_dir / "edge/runtime_cpp/visionops_runtime_mock"
    assert binary.is_file()
    return binary


@pytest.fixture(scope="session")
def shared_postprocess_fixture_binary(shared_runtime_build_dir: Path) -> Path:
    subprocess.run(
        [
            "cmake",
            "--build",
            str(shared_runtime_build_dir),
            "-j4",
            "--target",
            "visionops_postprocess_fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    binary = shared_runtime_build_dir / "edge/runtime_cpp/visionops_postprocess_fixture"
    assert binary.is_file()
    return binary
