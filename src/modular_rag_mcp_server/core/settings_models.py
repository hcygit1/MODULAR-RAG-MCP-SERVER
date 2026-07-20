"""Typed configuration models used by the settings loader."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMConfig:
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
    provider: str
    model: str
    openai_api_key: str = ""
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com"
    local_model_path: str = ""
    device: str = "cpu"


@dataclass
class VectorStoreConfig:
    backend: str
    persist_path: str
    collection_name: str
    sqlite_path: str = "./data/db/rag.sqlite"
    embedding_dim: int = 1536
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""


@dataclass
class RetrievalConfig:
    sparse_backend: str
    fusion_algorithm: str
    top_k_dense: int
    top_k_sparse: int
    top_k_final: int
    aggregate_by_parent: bool = False
    parent_id_metadata_key: str = "parent_id"
    parent_score_strategy: str = "sum"
    parent_aggregate_top_m: int = 30
    top_k_dense_parent: Optional[int] = None
    top_k_sparse_parent: Optional[int] = None


@dataclass
class RerankConfig:
    backend: str
    model: str
    top_m: int
    timeout_seconds: int


@dataclass
class EvaluationConfig:
    backends: List[str]
    golden_test_set: str


@dataclass
class LoggingConfig:
    log_file: str
    log_level: str


@dataclass
class DashboardConfig:
    enabled: bool
    port: int


@dataclass
class ObservabilityConfig:
    enabled: bool
    logging: LoggingConfig
    detail_level: str
    dashboard: DashboardConfig


@dataclass
class MinerUConfig:
    api_token: str = ""
    model_version: str = "vlm"
    poll_interval_seconds: int = 5
    poll_timeout_seconds: int = 600


@dataclass
class IngestionConfig:
    chunk_size: int
    chunk_overlap: int
    enable_llm_refinement: bool
    enable_metadata_enrichment: bool
    enable_image_captioning: bool
    batch_size: int
    splitter_strategy: str = "recursive"
    heading_split_level: int = 3
    heading_parent_level: int = 2
    enable_boundary_repair: bool = False
    boundary_repair_mode: str = "rule"


@dataclass
class Settings:
    llm: LLMConfig
    vision_llm: VisionLLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    evaluation: EvaluationConfig
    observability: ObservabilityConfig
    ingestion: IngestionConfig
    mineru: MinerUConfig = field(default_factory=MinerUConfig)
