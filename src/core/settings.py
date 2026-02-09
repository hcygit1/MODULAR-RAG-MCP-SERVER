"""
配置加载与校验模块

负责读取 config/settings.yaml 并解析为类型安全的 Settings 对象。
提供配置校验功能，确保必填字段存在。
"""
import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str
    model: str
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_api_version: str = "2024-02-15-preview"
    deployment_name: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com"


@dataclass
class VisionLLMConfig:
    """Vision LLM 配置"""
    provider: str
    model: str
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_api_version: str = "2024-02-15-preview"
    deployment_name: str = ""
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com"


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    provider: str
    model: str
    openai_api_key: str = ""
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com"
    local_model_path: str = ""
    device: str = "cpu"


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    backend: str
    persist_path: str
    collection_name: str


@dataclass
class RetrievalConfig:
    """检索配置"""
    sparse_backend: str
    fusion_algorithm: str
    top_k_dense: int
    top_k_sparse: int
    top_k_final: int


@dataclass
class RerankConfig:
    """重排配置"""
    backend: str
    model: str
    top_m: int
    timeout_seconds: int


@dataclass
class EvaluationConfig:
    """评估配置"""
    backends: List[str]
    golden_test_set: str


@dataclass
class LoggingConfig:
    """日志配置"""
    log_file: str
    log_level: str


@dataclass
class DashboardConfig:
    """Dashboard 配置"""
    enabled: bool
    port: int


@dataclass
class ObservabilityConfig:
    """可观测性配置"""
    enabled: bool
    logging: LoggingConfig
    detail_level: str
    dashboard: DashboardConfig


@dataclass
class IngestionConfig:
    """Ingestion Pipeline 配置"""
    chunk_size: int
    chunk_overlap: int
    enable_llm_refinement: bool
    enable_metadata_enrichment: bool
    enable_image_captioning: bool
    batch_size: int


@dataclass
class Settings:
    """主配置类，包含所有子系统配置"""
    llm: LLMConfig
    vision_llm: VisionLLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    evaluation: EvaluationConfig
    observability: ObservabilityConfig
    ingestion: IngestionConfig


def _resolve_env_vars(value: str) -> str:
    """解析环境变量占位符，例如 ${VAR_NAME}"""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.getenv(env_var, "")
    return value


def _load_yaml(path: str) -> dict:
    """加载 YAML 文件"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if data is None:
        raise ValueError(f"配置文件为空或格式错误: {path}")
    
    return data


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
    )
    
    retrieval_data = data.get("retrieval", {})
    retrieval = RetrievalConfig(
        sparse_backend=retrieval_data.get("sparse_backend", ""),
        fusion_algorithm=retrieval_data.get("fusion_algorithm", ""),
        top_k_dense=retrieval_data.get("top_k_dense", 20),
        top_k_sparse=retrieval_data.get("top_k_sparse", 20),
        top_k_final=retrieval_data.get("top_k_final", 10),
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
    if not settings.vector_store.persist_path:
        errors.append("vector_store.persist_path")
    if not settings.vector_store.collection_name:
        errors.append("vector_store.collection_name")
    
    # 校验 Retrieval 配置
    if not settings.retrieval.sparse_backend:
        errors.append("retrieval.sparse_backend")
    if not settings.retrieval.fusion_algorithm:
        errors.append("retrieval.fusion_algorithm")
    
    # 校验 Rerank 配置
    if not settings.rerank.backend:
        errors.append("rerank.backend")
    if settings.rerank.backend != "none" and not settings.rerank.model:
        errors.append("rerank.model")
    
    # 校验 Evaluation 配置
    if not settings.evaluation.backends:
        errors.append("evaluation.backends")
    
    # 校验 Observability 配置
    if not settings.observability.logging.log_file:
        errors.append("observability.logging.log_file")
    
    if errors:
        error_msg = "缺失必填字段: " + ", ".join(errors)
        raise ValueError(error_msg)


def load_settings(path: str = "config/settings.yaml") -> Settings:
    """
    加载配置文件并返回 Settings 对象
    
    Args:
        path: 配置文件路径，默认为 "config/settings.yaml"
        
    Returns:
        Settings: 解析后的配置对象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式错误或必填字段缺失
    """
    # 加载 YAML
    data = _load_yaml(path)
    
    # 解析为 Settings 对象
    settings = _parse_config(data)
    
    # 校验必填字段
    validate_settings(settings)
    
    return settings
