"""
Ingestion Pipeline 模块

提供文档摄取、处理、存储的完整流程。
实现 integrity→load→split→transform→boundary_repair→encode→store 的完整编排。
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.ingestion.document_chunk_builder import DocumentChunkBuilder
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.encoding_runner import EncodingRunner
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.models import Chunk, Document
from src.ingestion.storage.storage_coordinator import StorageCoordinator
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.transform.boundary_repairer import BoundaryRepairer
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.transform_runner import TransformRunner
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    build_processing_profile,
)
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Ingestion Pipeline 类

    负责将文档通过完整的流程处理：
    integrity → load → split → transform → encode → store

    串行执行各个步骤，并对失败步骤做清晰异常处理。
    """

    def __init__(
        self,
        settings: Settings,
        splitter_strategy: Optional[str] = None,
        llm: Optional[Any] = None,
        vision_llm: Optional[Any] = None,
        embedding: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        integrity_checker: Optional[FileIntegrityChecker] = None,
    ):
        """
        初始化 Ingestion Pipeline

        Args:
            settings: 配置对象，包含所有组件的配置
            splitter_strategy: Splitter 策略（可选），如果不提供则使用默认策略
            llm: LLM 实例（可选），用于 Transform 阶段的 LLM 增强
            vision_llm: Vision LLM 实例（可选），用于图片描述生成
            embedding: Embedding 实例（可选），如果不提供则从 settings 创建
            vector_store: VectorStore 实例（可选），如果不提供则从 settings 创建
            integrity_checker: 文件完整性检查器（可选），如果不提供则使用默认数据库
        """
        self._settings = settings
        self._ingestion_config = settings.ingestion

        # 初始化各个组件
        self._integrity_checker = (
            integrity_checker
            if integrity_checker is not None
            else FileIntegrityChecker()
        )
        splitter = SplitterFactory.create(settings, strategy=splitter_strategy)
        self._chunk_builder = DocumentChunkBuilder(splitter)

        # Text LLM：仅在未传入且 (enable_llm_refinement 或 enable_metadata_enrichment) 时创建
        if llm is None and (
            self._ingestion_config.enable_llm_refinement
            or self._ingestion_config.enable_metadata_enrichment
        ):
            try:
                from src.libs.llm.llm_factory import LLMFactory
                llm = LLMFactory.create(config=settings.llm)
            except (ValueError, NotImplementedError):
                llm = None

        # Transform 组件
        chunk_refiner = ChunkRefiner(
            config=self._ingestion_config,
            llm=llm
        )
        metadata_enricher = MetadataEnricher(
            config=self._ingestion_config,
            llm=llm
        )

        # 统一存储：图片经 VectorStore 写入，不再使用 ImageStorage 文件写入
        # ImageCaptioner 从 chunk metadata 的 image_data 获取图片（回退路径）
        # Vision LLM：仅在未传入且 enable_image_captioning 时创建
        if vision_llm is None and self._ingestion_config.enable_image_captioning:
            from src.libs.llm.llm_factory import LLMFactory
            vision_llm = LLMFactory.create_vision_llm(settings)

        image_captioner = ImageCaptioner(
            config=self._ingestion_config,
            vision_llm_config=settings.vision_llm,
            vision_llm=vision_llm,
            image_storage=None
        )

        # Boundary Repair 组件（可选，默认关闭）
        boundary_repairer: Optional[BoundaryRepairer] = None
        if self._ingestion_config.enable_boundary_repair:
            boundary_repairer = BoundaryRepairer(
                config=self._ingestion_config,
                llm=llm,
            )
        self._transform_runner = TransformRunner(
            chunk_refiner=chunk_refiner,
            metadata_enricher=metadata_enricher,
            image_captioner=image_captioner,
            boundary_repairer=boundary_repairer,
        )

        # Embedding 组件
        if embedding is None:
            embedding = EmbeddingFactory.create(settings)
        self._dense_encoder = DenseEncoder(
            embedding=embedding,
            batch_size=None  # BatchProcessor 负责批处理
        )
        self._sparse_encoder = SparseEncoder()
        batch_processor = BatchProcessor(
            dense_encoder=self._dense_encoder,
            sparse_encoder=self._sparse_encoder,
            batch_size=self._ingestion_config.batch_size
        )
        self._encoding_runner = EncodingRunner(batch_processor)

        # Storage 组件（统一存储：chunk+vec+sparse 由 VectorStore 完成，不再使用独立 BM25Indexer）
        if vector_store is None:
            vector_store = VectorStoreFactory.create(settings)
        vector_upserter = VectorUpserter(vector_store=vector_store)
        self._storage_coordinator = StorageCoordinator(vector_upserter)

    @property
    def _splitter(self) -> Any:
        return self._chunk_builder.splitter

    @_splitter.setter
    def _splitter(self, value: Any) -> None:
        self._chunk_builder.splitter = value

    @property
    def _chunk_refiner(self) -> Any:
        return self._transform_runner.chunk_refiner

    @_chunk_refiner.setter
    def _chunk_refiner(self, value: Any) -> None:
        self._transform_runner.chunk_refiner = value

    @property
    def _metadata_enricher(self) -> Any:
        return self._transform_runner.metadata_enricher

    @_metadata_enricher.setter
    def _metadata_enricher(self, value: Any) -> None:
        self._transform_runner.metadata_enricher = value

    @property
    def _image_captioner(self) -> Any:
        return self._transform_runner.image_captioner

    @_image_captioner.setter
    def _image_captioner(self, value: Any) -> None:
        self._transform_runner.image_captioner = value

    @property
    def _boundary_repairer(self) -> Optional[Any]:
        return self._transform_runner.boundary_repairer

    @_boundary_repairer.setter
    def _boundary_repairer(self, value: Optional[Any]) -> None:
        self._transform_runner.boundary_repairer = value

    @property
    def _batch_processor(self) -> Any:
        return self._encoding_runner.batch_processor

    @_batch_processor.setter
    def _batch_processor(self, value: Any) -> None:
        self._encoding_runner.batch_processor = value

    @property
    def _vector_upserter(self) -> Any:
        return self._storage_coordinator.vector_upserter

    @_vector_upserter.setter
    def _vector_upserter(self, value: Any) -> None:
        self._storage_coordinator.vector_upserter = value

    def close(self) -> None:
        """关闭资源（如 VectorStore、FTS5 连接），避免进程退出时析构报错"""
        store = self._vector_upserter.get_vector_store()
        if store is not None:
            store.close()

    def process(
        self,
        file_path: str,
        collection_name: str,
        trace: Optional[Any] = None,
        force: bool = False,
    ) -> None:
        """
        处理单个文件，执行完整的 ingestion 流程

        Args:
            file_path: 文件路径
            collection_name: 集合名称，用于组织存储
            trace: 追踪上下文（可选）
            force: 是否强制重新处理，为 True 时跳过完整性检查（忽略已处理记录）

        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当参数无效时
            RuntimeError: 当处理步骤失败时（会包含清晰的错误信息）
        """
        if not file_path:
            raise ValueError("file_path 不能为空")

        if not collection_name:
            raise ValueError("collection_name 不能为空")

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if trace is None:
            trace = TraceContext(operation="ingestion")
        _trace: Optional[TraceContext] = trace if isinstance(trace, TraceContext) else None

        if _trace:
            _trace.set_metric("file_path", file_path)
            _trace.set_metric("collection_name", collection_name)

        # 步骤 1: Integrity Check（文件完整性检查，force 时跳过）
        try:
            processing_profile = build_processing_profile(
                self._settings,
                loader_name="pdf",
            )
            if _trace:
                with _trace.stage("integrity_check"):
                    file_hash = self._integrity_checker.compute_sha256(file_path)
                    should_skip = (
                        False
                        if force
                        else self._integrity_checker.should_skip(
                            file_hash,
                            collection_name,
                            processing_profile,
                        )
                    )
            else:
                file_hash = self._integrity_checker.compute_sha256(file_path)
                should_skip = (
                    False
                    if force
                    else self._integrity_checker.should_skip(
                        file_hash,
                        collection_name,
                        processing_profile,
                    )
                )
            if should_skip:
                if _trace:
                    _trace.set_metric("skipped", True)
                return
        except Exception as e:
            raise RuntimeError(f"文件完整性检查失败: {str(e)}") from e

        # 步骤 2: Load（加载文档）
        try:
            if _trace:
                with _trace.stage("load"):
                    loader = PdfLoader()
                    document = loader.load(file_path, trace=trace)
            else:
                loader = PdfLoader()
                document = loader.load(file_path, trace=trace)
        except Exception as e:
            raise RuntimeError(f"文档加载失败: {str(e)}") from e

        self.process_document(document, collection_name, trace=trace)

        # 标记处理成功（失败不影响主流程，但需记录日志避免重复处理未被发现）
        try:
            self._integrity_checker.mark_success(
                file_hash,
                collection_name,
                processing_profile,
            )
        except Exception as e:
            logger.warning(
                "标记文件处理成功失败 (file_hash=%s)，可能导致后续重复处理: %s",
                file_hash[:16] if file_hash else "?",
                e,
            )

        if _trace:
            _trace.set_metric("skipped", False)

    def process_document(
        self,
        document: Document,
        collection_name: str,
        trace: Optional[Any] = None,
    ) -> int:
        """
        从已有 Document 执行 Split → Transform → Encode → Store（跳过 integrity、load）

        用于 MinerU 或普通解析工具产出的 Document 直接入库。

        Args:
            document: 已构建的 Document 对象
            collection_name: 集合名称
            trace: 追踪上下文（可选）

        Returns:
            int: 写入的 chunk 数量

        Raises:
            ValueError: Document 无效
            RuntimeError: 处理步骤失败
        """
        if not document or not document.text:
            raise ValueError("Document 或 Document.text 不能为空")
        if not collection_name or not str(collection_name).strip():
            raise ValueError("collection_name 不能为空")

        if trace is None:
            trace = TraceContext(operation="ingestion")
        _trace: Optional[TraceContext] = trace if isinstance(trace, TraceContext) else None

        if _trace:
            _trace.set_metric("doc_id", document.id)
            _trace.set_metric("collection_name", collection_name)

        # 步骤 1: Split
        try:
            if _trace:
                with _trace.stage("split"):
                    chunks = self.split_document(document, trace=trace)
                _trace.set_metric("chunk_count", len(chunks))
            else:
                chunks = self.split_document(document, trace=trace)
            if not chunks:
                raise RuntimeError("文档切分后未产生任何 chunks")
        except Exception as e:
            raise RuntimeError(f"文档切分失败: {str(e)}") from e

        return self.process_chunks(chunks, collection_name, trace=trace)

    def process_chunks(
        self,
        chunks: List[Chunk],
        collection_name: str,
        trace: Optional[Any] = None,
    ) -> int:
        """
        从已有 Chunk 执行 Transform → Encode → Store（跳过 load 和 split）。

        用于 Docling / MinerU 这类已经完成结构化切块的入口，避免表格和图片
        block 被普通 splitter 再次切坏。

        Args:
            chunks: 已切好的 Chunk 列表
            collection_name: 集合名称
            trace: 追踪上下文（可选）

        Returns:
            int: 写入的 chunk 数量
        """
        if not chunks:
            raise ValueError("chunks 不能为空")
        if not collection_name or not str(collection_name).strip():
            raise ValueError("collection_name 不能为空")

        if trace is None:
            trace = TraceContext(operation="ingestion")
        _trace: Optional[TraceContext] = trace if isinstance(trace, TraceContext) else None

        if _trace:
            _trace.set_metric("collection_name", collection_name)
            _trace.set_metric("chunk_count", len(chunks))
            source_doc_id = str(chunks[0].metadata.get("source_doc_id", "")).strip()
            if source_doc_id:
                _trace.set_metric("doc_id", source_doc_id)

        # 步骤 3: Transform
        try:
            if _trace:
                with _trace.stage("transform"):
                    transformed_chunks = self._apply_transforms(chunks, trace=trace)
            else:
                transformed_chunks = self._apply_transforms(chunks, trace=trace)
        except Exception as e:
            raise RuntimeError(f"Chunk 转换失败: {str(e)}") from e

        # 步骤 3.5: Boundary Repair（可选，修复块边界断裂）
        if self._boundary_repairer is not None:
            stage_context = (
                _trace.stage("boundary_repair")
                if _trace
                else None
            )
            transformed_chunks = self._transform_runner.repair_boundaries(
                transformed_chunks,
                trace=trace,
                stage_context=stage_context,
            )

        # 步骤 4: Encode
        try:
            if _trace:
                with _trace.stage("encode"):
                    dense_vectors, sparse_vectors = self._encoding_runner.encode(
                        transformed_chunks, trace=trace
                    )
            else:
                dense_vectors, sparse_vectors = self._encoding_runner.encode(
                    transformed_chunks, trace=trace
                )
        except Exception as e:
            raise RuntimeError(f"向量编码失败: {str(e)}") from e

        # 步骤 5: Store
        try:
            if _trace:
                with _trace.stage("store"):
                    self._store_results(
                        transformed_chunks, dense_vectors, sparse_vectors,
                        collection_name, trace=trace,
                    )
            else:
                self._store_results(
                    transformed_chunks, dense_vectors, sparse_vectors,
                    collection_name, trace=trace,
                )
        except Exception as e:
            raise RuntimeError(f"存储失败: {str(e)}") from e

        return len(transformed_chunks)

    def _apply_transforms(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None
    ) -> List[Chunk]:
        """
        应用所有 Transform 组件

        Args:
            chunks: 输入的 Chunk 列表
            trace: 追踪上下文（可选）

        Returns:
            List[Chunk]: 转换后的 Chunk 列表
        """
        return self._transform_runner.run(chunks, trace=trace)

    def _store_results(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, float]],
        collection_name: str,
        trace: Optional[Any] = None
    ) -> None:
        """
        存储处理结果。若 BM25 步骤失败，会回滚已写入的向量以保持 collection 一致性。

        Args:
            chunks: Chunk 列表
            dense_vectors: 稠密向量列表
            sparse_vectors: 稀疏向量列表
            collection_name: 集合名称
            trace: 追踪上下文（可选）
        """
        self._storage_coordinator.store(
            chunks,
            dense_vectors,
            sparse_vectors,
            collection_name,
            trace=trace,
        )

    def split_document(
        self,
        document: Document,
        trace: Optional[Any] = None
    ) -> List[Chunk]:
        """
        将 Document 切分为多个 Chunk

        使用配置的 Splitter 将文档文本切分为多个片段，每个片段包含：
        - 文本内容
        - 元数据（继承自 Document，并添加 chunk 特定信息）
        - 位置信息（start_offset, end_offset）

        Args:
            document: 要切分的 Document 对象
            trace: 追踪上下文（可选），用于记录性能指标和调试信息

        Returns:
            List[Chunk]: 切分后的 Chunk 列表

        Raises:
            ValueError: 当 Document 无效时
            RuntimeError: 当切分过程失败时
        """
        return self._chunk_builder.build(document, trace=trace)

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """
        生成 Chunk 唯一标识符

        Args:
            doc_id: 文档 ID
            chunk_index: Chunk 索引

        Returns:
            str: Chunk ID
        """
        return self._chunk_builder.generate_chunk_id(doc_id, chunk_index)

    def _extract_image_refs(self, text: str) -> List[str]:
        """
        从文本中提取图片引用（占位符）

        提取格式为 [IMAGE: {image_id}] 的占位符。

        Args:
            text: 文本内容

        Returns:
            List[str]: 图片 ID 列表
        """
        return self._chunk_builder.extract_image_refs(text)

    def get_splitter_strategy(self) -> str:
        """
        获取当前使用的 Splitter 策略

        Returns:
            str: 策略名称
        """
        return self._chunk_builder.get_splitter_strategy()

    def get_chunk_size(self) -> int:
        """
        获取当前配置的块大小

        Returns:
            int: 块大小
        """
        return self._chunk_builder.get_chunk_size()

    def get_chunk_overlap(self) -> int:
        """
        获取当前配置的块重叠大小

        Returns:
            int: 重叠大小
        """
        return self._chunk_builder.get_chunk_overlap()
