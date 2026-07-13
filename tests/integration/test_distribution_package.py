import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PACKAGE_FILES = {
    "src/core/settings.py",
    "src/ingestion/pipeline.py",
    "src/libs/vector_store/base_vector_store.py",
    "src/mcp_server/server.py",
    "src/observability/logger.py",
}


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source_dir = tmp_path_factory.mktemp("source") / "project"
    source_dir.mkdir()
    shutil.copy2(PROJECT_ROOT / "pyproject.toml", source_dir / "pyproject.toml")
    shutil.copy2(PROJECT_ROOT / "README.md", source_dir / "README.md")
    shutil.copytree(
        PROJECT_ROOT / "src",
        source_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    wheel_dir = tmp_path_factory.mktemp("wheel")
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(source_dir),
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=wheel_dir,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_wheel_contains_application_subpackages(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as archive:
        names = set(archive.namelist())

    missing = EXPECTED_PACKAGE_FILES - names
    assert not missing, f"wheel 缺少业务模块: {sorted(missing)}"
    assert not any(name.startswith(("tests/", "scripts/")) for name in names)


def test_wheel_imports_mcp_server_outside_repository(
    built_wheel: Path,
    tmp_path: Path,
) -> None:
    install_dir = tmp_path / "installed"
    with zipfile.ZipFile(built_wheel) as archive:
        archive.extractall(install_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(install_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "import src.mcp_server.server as server; "
                "print(f'IMPORTED_FROM={Path(server.__file__).resolve()}')"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    marker = "IMPORTED_FROM="
    imported_line = next(
        (line for line in result.stdout.splitlines() if line.startswith(marker)),
        None,
    )
    assert imported_line is not None, result.stdout

    imported_from = Path(imported_line.removeprefix(marker))
    assert imported_from.is_relative_to(install_dir)
