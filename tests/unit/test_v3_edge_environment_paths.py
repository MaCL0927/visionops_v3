from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_VENV = "/opt/visionops/venv"


def test_edge_launchers_and_env_templates_do_not_default_to_v2_venv() -> None:
    paths = [
        ROOT / "scripts/start_collector.sh",
        ROOT / "scripts/start_runtime.sh",
        *sorted((ROOT / "production/carton_line/scripts").glob("*.sh")),
        ROOT / "production/carton_line/deploy/production.env.example",
        *sorted((ROOT / "production/carton_palletizing/scripts").glob("*.sh")),
        ROOT / "production/carton_palletizing/deploy/production.env.example",
    ]
    offenders = [
        str(path.relative_to(ROOT))
        for path in paths
        if LEGACY_VENV in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_installers_migrate_existing_v2_venv_setting() -> None:
    installers = [
        ROOT / "production/carton_line/deploy/install_services.sh",
        ROOT / "production/carton_palletizing/deploy/install_services.sh",
        ROOT / "production/carton_palletizing/deploy/install_box_grasp_services.sh",
    ]
    for path in installers:
        text = path.read_text(encoding="utf-8")
        assert f"^VISIONOPS_VENV={LEGACY_VENV}$" in text
        assert "VISIONOPS_VENV=${ROOT}/venv" in text


def test_edge_environment_script_uses_v3_venv() -> None:
    text = (ROOT / "scripts/setup_edge_env.sh").read_text(encoding="utf-8")
    assert 'VENV="${VISIONOPS_VENV:-${ROOT}/venv}"' in text
    assert "python3 -m venv --system-site-packages" in text
    assert "requirements/edge-runtime.txt" in text


def test_edge_environment_verifier_imports_real_config_symbols() -> None:
    text = (ROOT / "scripts/setup_edge_env.sh").read_text(encoding="utf-8")
    assert "import load_line_config" not in text
    assert "load_config as load_carton_line_config" in text
    assert "load_config as load_palletizing_config" in text
    assert "--verify-only" in text


def test_installer_venv_migration_sed_is_shell_safe(tmp_path) -> None:
    import subprocess

    installers = [
        ROOT / "production/carton_line/deploy/install_services.sh",
        ROOT / "production/carton_palletizing/deploy/install_services.sh",
        ROOT / "production/carton_palletizing/deploy/install_box_grasp_services.sh",
    ]

    for index, path in enumerate(installers):
        text = path.read_text(encoding="utf-8")
        sed_line = next(
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith("sed -i ") and "VISIONOPS_VENV=" in line
        )

        env_file = tmp_path / f"installer-{index}.env"
        env_file.write_text(
            "VISIONOPS_VENV=/opt/visionops/venv\n",
            encoding="utf-8",
        )
        command = (
            f'ROOT=/opt/visionops_v3; ENV_FILE="{env_file}"; '
            f"{sed_line}"
        )
        subprocess.run(["bash", "-c", command], check=True)

        assert env_file.read_text(encoding="utf-8") == (
            "VISIONOPS_VENV=/opt/visionops_v3/venv\n"
        )
