"""E2E 测试的隔离运行环境。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def e2e_runtime(tmp_path):
    """生成完全离线、不会读写项目业务数据库的 E2E 配置。"""
    sqlite_path = tmp_path / "rag.sqlite"
    config_path = tmp_path / "settings.e2e.yaml"
    config = {
        "llm": {
            "provider": "ollama",
            "model": "e2e-unused",
            "ollama_base_url": "http://localhost:11434",
        },
        "vision_llm": {
            "provider": "qwen",
            "model": "e2e-unused",
        },
        "embedding": {
            "provider": "local",
            "model": "bge-base-zh-v1.5",
            "local_model_path": "",
            "device": "cpu",
        },
        "vector_store": {
            "backend": "sqlite",
            "persist_path": str(tmp_path / "chroma"),
            "collection_name": "test_collection",
            "sqlite_path": str(sqlite_path),
            "embedding_dim": 768,
        },
        "retrieval": {
            "sparse_backend": "fts5",
            "fusion_algorithm": "rrf",
            "top_k_dense": 10,
            "top_k_sparse": 10,
            "top_k_final": 5,
        },
        "rerank": {
            "backend": "none",
            "model": "",
            "top_m": 5,
            "timeout_seconds": 5,
        },
        "evaluation": {
            "backends": ["custom"],
            "golden_test_set": str(
                PROJECT_ROOT / "tests" / "fixtures" / "golden_test_set.json"
            ),
        },
        "observability": {
            "enabled": False,
            "logging": {
                "log_file": str(tmp_path / "traces.jsonl"),
                "log_level": "INFO",
            },
            "detail_level": "minimal",
            "dashboard": {
                "enabled": False,
                "port": 8501,
            },
        },
        "ingestion": {
            "chunk_size": 256,
            "chunk_overlap": 25,
            "enable_llm_refinement": False,
            "enable_metadata_enrichment": False,
            "enable_image_captioning": False,
            "batch_size": 8,
            "splitter_strategy": "recursive",
            "heading_split_level": 3,
            "heading_parent_level": 2,
            "enable_boundary_repair": False,
            "boundary_repair_mode": "rule",
        },
    }
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PROJECT_ROOT / "src"), python_path) if part
    )
    env["MODULAR_RAG_CONFIG_PATH"] = str(config_path)

    return {
        "config_path": config_path,
        "env": env,
        "project_root": PROJECT_ROOT,
        "sqlite_path": sqlite_path,
        "workdir": tmp_path,
    }


@pytest.fixture
def sample_pdf_path():
    """返回仓库内稳定的样例 PDF。"""
    pdf_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip("sample.pdf 不存在，跳过测试")
    return pdf_path


@pytest.fixture
def ingested_e2e_runtime(e2e_runtime, sample_pdf_path):
    """在隔离库中预先摄取样例 PDF，供 MCP 查询测试使用。"""
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ingest.py"),
            "--collection",
            "test_collection",
            "--path",
            str(sample_pdf_path),
            "--config",
            str(e2e_runtime["config_path"]),
        ],
        capture_output=True,
        text=True,
        cwd=e2e_runtime["workdir"],
        env=e2e_runtime["env"],
    )
    assert result.returncode == 0, result.stderr
    return e2e_runtime
