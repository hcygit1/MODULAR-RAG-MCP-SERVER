"""
配置加载与校验模块

负责读取包内默认配置或外部 YAML，并解析为类型安全的 Settings 对象。
提供配置校验功能，确保必填字段存在。
"""
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from modular_rag_mcp_server.core.runtime_paths import resolve_runtime_path
from modular_rag_mcp_server.core.settings_models import (
    DashboardConfig,
    EmbeddingConfig,
    EvaluationConfig,
    IngestionConfig,
    LLMConfig,
    LoggingConfig,
    MinerUConfig,
    ObservabilityConfig,
    RerankConfig,
    RetrievalConfig,
    Settings,
    VectorStoreConfig,
    VisionLLMConfig,
)
from modular_rag_mcp_server.package_resources import read_resource_text

_ALLOWED_TOP_LEVEL_KEYS = {
    "llm",
    "vision_llm",
    "embedding",
    "vector_store",
    "retrieval",
    "rerank",
    "evaluation",
    "observability",
    "ingestion",
    "mineru",
}


def _resolve_env_vars(value: str) -> str:
    """解析环境变量占位符，例如 ${VAR_NAME}"""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.getenv(env_var, "")
    return value


def _parse_yaml(text: str, source: str) -> dict:
    """解析一个 YAML 配置来源。"""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"配置文件格式错误: {source}") from exc
    if data is None:
        raise ValueError(f"配置文件为空或格式错误: {source}")
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射: {source}")
    return data


def _load_yaml(path: str) -> dict:
    """加载 YAML 文件。"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    return _parse_yaml(config_path.read_text(encoding="utf-8"), path)


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并，override 中的值覆盖 base。仅处理 dict 与标量，不改动 base 中未被 override 的键。"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_config(data: dict) -> Settings:
    """解析配置字典为 Settings 对象"""
    # 解析嵌套配置
    llm_data = data.get("llm", {})
    llm = LLMConfig(
        provider=llm_data.get("provider", ""),
        model=llm_data.get("model", ""),
        azure_endpoint=_resolve_env_vars(llm_data.get("azure_endpoint", "")),
        azure_api_key=_resolve_env_vars(llm_data.get("azure_api_key", "")),
        azure_api_version=llm_data.get("azure_api_version", "2024-02-15-preview"),
        deployment_name=llm_data.get("deployment_name", ""),
        openai_api_key=_resolve_env_vars(llm_data.get("openai_api_key", "")),
        ollama_base_url=llm_data.get("ollama_base_url", "http://localhost:11434"),
        deepseek_api_key=_resolve_env_vars(llm_data.get("deepseek_api_key", "")),
        deepseek_base_url=llm_data.get("deepseek_base_url", "https://api.deepseek.com"),
        dashscope_api_key=_resolve_env_vars(llm_data.get("dashscope_api_key", "")),
        dashscope_base_url=llm_data.get("dashscope_base_url", "https://dashscope.aliyuncs.com"),
    )

    vision_llm_data = data.get("vision_llm", {})
    vision_llm = VisionLLMConfig(
        provider=vision_llm_data.get("provider", ""),
        model=vision_llm_data.get("model", ""),
        azure_endpoint=_resolve_env_vars(vision_llm_data.get("azure_endpoint", "")),
        azure_api_key=_resolve_env_vars(vision_llm_data.get("azure_api_key", "")),
        azure_api_version=vision_llm_data.get("azure_api_version", "2024-02-15-preview"),
        deployment_name=vision_llm_data.get("deployment_name", ""),
        dashscope_api_key=_resolve_env_vars(vision_llm_data.get("dashscope_api_key", "")),
        dashscope_base_url=vision_llm_data.get("dashscope_base_url", "https://dashscope.aliyuncs.com"),
    )

    embedding_data = data.get("embedding", {})
    embedding = EmbeddingConfig(
        provider=embedding_data.get("provider", ""),
        model=embedding_data.get("model", ""),
        openai_api_key=_resolve_env_vars(embedding_data.get("openai_api_key", "")),
        dashscope_api_key=_resolve_env_vars(embedding_data.get("dashscope_api_key", "")),
        dashscope_base_url=embedding_data.get("dashscope_base_url", "https://dashscope.aliyuncs.com"),
        local_model_path=embedding_data.get("local_model_path", ""),
        device=embedding_data.get("device", "cpu"),
    )

    vector_store_data = data.get("vector_store", {})
    vector_store = VectorStoreConfig(
        backend=vector_store_data.get("backend", ""),
        persist_path=vector_store_data.get("persist_path", ""),
        collection_name=vector_store_data.get("collection_name", ""),
        sqlite_path=vector_store_data.get("sqlite_path", "./data/db/rag.sqlite"),
        embedding_dim=int(vector_store_data.get("embedding_dim", 1536)),
        qdrant_url=_resolve_env_vars(vector_store_data.get("qdrant_url", "http://localhost:6333")),
        qdrant_api_key=_resolve_env_vars(vector_store_data.get("qdrant_api_key", "")),
    )

    retrieval_data = data.get("retrieval", {})
    retrieval = RetrievalConfig(
        sparse_backend=retrieval_data.get("sparse_backend", ""),
        fusion_algorithm=retrieval_data.get("fusion_algorithm", ""),
        top_k_dense=retrieval_data.get("top_k_dense", 20),
        top_k_sparse=retrieval_data.get("top_k_sparse", 20),
        top_k_final=retrieval_data.get("top_k_final", 10),
        aggregate_by_parent=retrieval_data.get("aggregate_by_parent", False),
        parent_id_metadata_key=retrieval_data.get("parent_id_metadata_key", "parent_id"),
        parent_score_strategy=retrieval_data.get("parent_score_strategy", "sum"),
        parent_aggregate_top_m=retrieval_data.get("parent_aggregate_top_m", 30),
        top_k_dense_parent=retrieval_data.get("top_k_dense_parent"),
        top_k_sparse_parent=retrieval_data.get("top_k_sparse_parent"),
    )

    rerank_data = data.get("rerank", {})
    rerank = RerankConfig(
        backend=rerank_data.get("backend", ""),
        model=rerank_data.get("model", ""),
        top_m=rerank_data.get("top_m", 30),
        timeout_seconds=rerank_data.get("timeout_seconds", 5),
    )

    evaluation_data = data.get("evaluation", {})
    evaluation = EvaluationConfig(
        backends=evaluation_data.get("backends", []),
        golden_test_set=evaluation_data.get("golden_test_set", ""),
    )

    observability_data = data.get("observability", {})
    logging_data = observability_data.get("logging", {})
    logging_config = LoggingConfig(
        log_file=logging_data.get("log_file", ""),
        log_level=logging_data.get("log_level", "INFO"),
    )
    dashboard_data = observability_data.get("dashboard", {})
    dashboard_config = DashboardConfig(
        enabled=dashboard_data.get("enabled", True),
        port=dashboard_data.get("port", 8501),
    )
    observability = ObservabilityConfig(
        enabled=observability_data.get("enabled", True),
        logging=logging_config,
        detail_level=observability_data.get("detail_level", "standard"),
        dashboard=dashboard_config,
    )

    ingestion_data = data.get("ingestion", {})
    ingestion = IngestionConfig(
        chunk_size=ingestion_data.get("chunk_size", 512),
        chunk_overlap=ingestion_data.get("chunk_overlap", 50),
        enable_llm_refinement=ingestion_data.get("enable_llm_refinement", False),
        enable_metadata_enrichment=ingestion_data.get("enable_metadata_enrichment", True),
        enable_image_captioning=ingestion_data.get("enable_image_captioning", True),
        batch_size=ingestion_data.get("batch_size", 32),
        splitter_strategy=ingestion_data.get("splitter_strategy", "recursive"),
        heading_split_level=ingestion_data.get("heading_split_level", 3),
        heading_parent_level=ingestion_data.get("heading_parent_level", 2),
        enable_boundary_repair=ingestion_data.get("enable_boundary_repair", False),
        boundary_repair_mode=ingestion_data.get("boundary_repair_mode", "rule"),
    )

    mineru_data = data.get("mineru", {})
    mineru = MinerUConfig(
        api_token=_resolve_env_vars(mineru_data.get("api_token", "")),
        model_version=mineru_data.get("model_version", "vlm"),
        poll_interval_seconds=mineru_data.get("poll_interval_seconds", 5),
        poll_timeout_seconds=mineru_data.get("poll_timeout_seconds", 600),
    )

    return Settings(
        llm=llm,
        vision_llm=vision_llm,
        embedding=embedding,
        vector_store=vector_store,
        retrieval=retrieval,
        rerank=rerank,
        evaluation=evaluation,
        observability=observability,
        ingestion=ingestion,
        mineru=mineru,
    )


def validate_settings(settings: Settings) -> None:
    """
    校验配置的必填字段

    Args:
        settings: Settings 对象

    Raises:
        ValueError: 当必填字段缺失时抛出，错误信息包含字段路径
    """
    errors = []

    ingestion = settings.ingestion
    if not isinstance(ingestion.chunk_size, int) or ingestion.chunk_size <= 0:
        errors.append("ingestion.chunk_size: 必须大于 0")
    if (
        not isinstance(ingestion.chunk_overlap, int)
        or ingestion.chunk_overlap < 0
        or ingestion.chunk_overlap >= ingestion.chunk_size
    ):
        errors.append("ingestion.chunk_overlap: 必须满足 0 <= overlap < chunk_size")
    if not isinstance(ingestion.batch_size, int) or ingestion.batch_size <= 0:
        errors.append("ingestion.batch_size: 必须大于 0")

    retrieval = settings.retrieval
    for field_name in ("top_k_dense", "top_k_sparse", "top_k_final", "parent_aggregate_top_m"):
        if not isinstance(getattr(retrieval, field_name), int) or getattr(retrieval, field_name) <= 0:
            errors.append(f"retrieval.{field_name}: 必须大于 0")

    if not isinstance(settings.rerank.top_m, int) or settings.rerank.top_m <= 0:
        errors.append("rerank.top_m: 必须大于 0")
    if not isinstance(settings.rerank.timeout_seconds, int) or settings.rerank.timeout_seconds < 0:
        errors.append("rerank.timeout_seconds: 不能为负数")
    dashboard_port = settings.observability.dashboard.port
    if not isinstance(dashboard_port, int) or not 1 <= dashboard_port <= 65535:
        errors.append("observability.dashboard.port: 必须在 1-65535 范围内")
    for field_name in ("poll_interval_seconds", "poll_timeout_seconds"):
        if not isinstance(getattr(settings.mineru, field_name), int) or getattr(settings.mineru, field_name) < 0:
            errors.append(f"mineru.{field_name}: 不能为负数")

    # 校验 LLM 配置
    if not settings.llm.provider:
        errors.append("llm.provider")
    if not settings.llm.model:
        errors.append("llm.model")

    # 校验 Vision LLM 配置
    if not settings.vision_llm.provider:
        errors.append("vision_llm.provider")
    if not settings.vision_llm.model:
        errors.append("vision_llm.model")

    # 校验 Embedding 配置
    if not settings.embedding.provider:
        errors.append("embedding.provider")
    if not settings.embedding.model:
        errors.append("embedding.model")

    # 校验 VectorStore 配置
    if not settings.vector_store.backend:
        errors.append("vector_store.backend")
    if settings.vector_store.backend.lower() == "sqlite":
        if not settings.vector_store.sqlite_path:
            errors.append("vector_store.sqlite_path")
        if not settings.vector_store.embedding_dim or settings.vector_store.embedding_dim <= 0:
            errors.append("vector_store.embedding_dim")
    elif not settings.vector_store.persist_path:
        errors.append("vector_store.persist_path")
    if not settings.vector_store.collection_name:
        errors.append("vector_store.collection_name")

    # 校验 Retrieval 配置
    if not settings.retrieval.sparse_backend:
        errors.append("retrieval.sparse_backend")
    if not settings.retrieval.fusion_algorithm:
        errors.append("retrieval.fusion_algorithm")

    # 统一存储约束：backend=sqlite 时必须 sparse_backend=fts5
    vs_backend = (settings.vector_store.backend or "").lower()
    sparse_backend = (settings.retrieval.sparse_backend or "").lower()
    if vs_backend == "sqlite" and sparse_backend != "fts5":
        errors.append(
            "retrieval.sparse_backend: backend=sqlite 时必须为 fts5 以实现统一存储"
        )

    # 弃用警告：sparse_backend=bm25（JSON BM25）为遗留模式
    if sparse_backend == "bm25":
        logging.warning(
            "sparse_backend=bm25 为遗留模式，推荐迁移至 "
            "backend=sqlite + sparse_backend=fts5 实现统一存储。"
            "参见 docs/SQLITE_AND_UNIFIED_STORAGE_PLAN.md"
        )

    # 弃用警告：chroma/qdrant 当前未实现统一存储（chunk+vec+BM25+图片同容器）
    if vs_backend in ("chroma", "qdrant"):
        logging.warning(
            "backend=%s 当前未满足统一存储约束（BM25/图片在独立容器）。"
            "推荐使用 backend=sqlite + sparse_backend=fts5。"
            "参见 docs/SQLITE_AND_UNIFIED_STORAGE_PLAN.md",
            vs_backend,
        )

    # 校验 Rerank 配置
    if not settings.rerank.backend:
        errors.append("rerank.backend")
    if settings.rerank.backend != "none" and not settings.rerank.model:
        errors.append("rerank.model")
    if (
        settings.retrieval.aggregate_by_parent
        and settings.rerank.backend != "none"
        and settings.rerank.top_m < settings.retrieval.parent_aggregate_top_m
    ):
        errors.append(
            "rerank.top_m: 父聚合模式下不能小于 "
            "retrieval.parent_aggregate_top_m"
        )

    # 校验 Evaluation 配置
    if not settings.evaluation.backends:
        errors.append("evaluation.backends")

    # 校验 Observability 配置
    if not settings.observability.logging.log_file:
        errors.append("observability.logging.log_file")

    if errors:
        prefix = "缺失必填字段" if all(": " not in error for error in errors) else "配置校验失败"
        error_msg = prefix + ": " + ", ".join(errors)
        raise ValueError(error_msg)


def load_settings(path: Optional[str] = None) -> Settings:
    """
    加载配置文件并返回 Settings 对象。
    若同目录下存在 settings.local.yaml，则对其做深度合并以覆盖敏感项等。

    Args:
        path: 可选主配置文件路径；未提供时读取 wheel 内置默认配置

    Returns:
        Settings: 解析后的配置对象

    Raises:
        FileNotFoundError: 主配置文件不存在
        ValueError: 配置文件格式错误或必填字段缺失
    """
    if path is None:
        data = _parse_yaml(
            read_resource_text("settings.yaml"),
            "package:resources/settings.yaml",
        )
        local_path = Path("config/settings.local.yaml")
    else:
        data = _load_yaml(path)
        local_path = Path(path).parent / "settings.local.yaml"

    unknown_keys = sorted(set(data) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown_keys:
        raise ValueError("未知配置字段: " + ", ".join(unknown_keys))

    # 工作目录或显式配置目录中的敏感项覆盖内置/基础配置。
    if local_path.exists():
        local_data = _load_yaml(str(local_path))
        unknown_local_keys = sorted(set(local_data) - _ALLOWED_TOP_LEVEL_KEYS)
        if unknown_local_keys:
            raise ValueError("未知配置字段: " + ", ".join(unknown_local_keys))
        data = _deep_merge(data, local_data)

    settings = _parse_config(data)
    settings.vector_store.sqlite_path = resolve_runtime_path(
        settings.vector_store.sqlite_path,
        "data",
    )
    settings.vector_store.persist_path = resolve_runtime_path(
        settings.vector_store.persist_path,
        "data",
    )
    settings.observability.logging.log_file = resolve_runtime_path(
        settings.observability.logging.log_file,
        "logs",
    )
    validate_settings(settings)
    return settings
