import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PACKAGE_FILES = {
    "modular_rag_mcp_server/core/settings.py",
    "modular_rag_mcp_server/ingestion/pipeline.py",
    "modular_rag_mcp_server/libs/vector_store/base_vector_store.py",
    "modular_rag_mcp_server/mcp_server/server.py",
    "modular_rag_mcp_server/observability/logger.py",
    "modular_rag_mcp_server/resources/settings.yaml",
    "modular_rag_mcp_server/resources/prompts/boundary_repair.txt",
    "modular_rag_mcp_server/resources/prompts/chunk_refinement.txt",
    "modular_rag_mcp_server/resources/prompts/image_captioning.txt",
    "modular_rag_mcp_server/resources/prompts/rerank.txt",
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
    assert not any(name == "src/__init__.py" or name.startswith("src/") for name in names)
    assert not any(name.startswith(("tests/", "scripts/")) for name in names)


def test_pytest_is_only_a_development_dependency(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode("utf-8")

    pytest_requirements = [
        line
        for line in metadata.splitlines()
        if line.lower().startswith("requires-dist: pytest")
    ]
    assert pytest_requirements
    assert all('extra == "dev"' in line for line in pytest_requirements)


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
                "import modular_rag_mcp_server.mcp_server.server as server; "
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


def test_wheel_installs_cli_and_loads_resources_outside_repository(
    built_wheel: Path,
    tmp_path: Path,
) -> None:
    install_dir = tmp_path / "site-packages"
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_dir),
            str(built_wheel),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    isolated_import = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import importlib.util, sys; "
                f"sys.path.insert(0, {str(install_dir)!r}); "
                "import modular_rag_mcp_server; "
                "assert importlib.util.find_spec('src') is None"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert isolated_import.returncode == 0, (
        isolated_import.stdout + isolated_import.stderr
    )

    script_name = "modular-rag-mcp-server.exe" if os.name == "nt" else (
        "modular-rag-mcp-server"
    )
    cli_path = install_dir / "bin" / script_name
    assert cli_path.is_file()
    cli_env = os.environ.copy()
    cli_env.pop("MODULAR_RAG_CONFIG_PATH", None)
    cli_env["PYTHONPATH"] = str(install_dir)
    cli = subprocess.run(
        [str(cli_path), "--check-config"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=cli_env,
        check=False,
    )
    assert cli.returncode == 0, cli.stdout + cli.stderr
