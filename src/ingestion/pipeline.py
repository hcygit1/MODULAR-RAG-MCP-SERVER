"""
Ingestion Pipeline 模块

提供文档摄取、处理、存储的完整流程。
实现 integrity→load→split→transform→encode→store 的完整编排。
"""
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict

from src.ingestion.models import Document, Chunk
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.loader.file_integrity import FileIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext

from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.storage.vector_upserter import VectorUpserter

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
        vector_store: Optional[Any] = None
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
        """
        self._settings = settings
        self._ingestion_config = settings.ingestion
        
        # 初始化各个组件
        self._integrity_checker = FileIntegrityChecker()
        self._splitter = SplitterFactory.create(settings, strategy=splitter_strategy)
        
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
        self._chunk_refiner = ChunkRefiner(
            config=self._ingestion_config,
            llm=llm
        )
        self._metadata_enricher = MetadataEnricher(
            config=self._ingestion_config,
            llm=llm
        )
        
        # 统一存储：图片经 VectorStore 写入，不再使用 ImageStorage 文件写入
        # ImageCaptioner 从 chunk metadata 的 image_data 获取图片（回退路径）
        # Vision LLM：仅在未传入且 enable_image_captioning 时创建
        if vision_llm is None and self._ingestion_config.enable_image_captioning:
            from src.libs.llm.llm_factory import LLMFactory
            vision_llm = LLMFactory.create_vision_llm(settings)
        
        self._image_captioner = ImageCaptioner(
            config=self._ingestion_config,
            vision_llm_config=settings.vision_llm,
            vision_llm=vision_llm,
            image_storage=None
        )
        
        # Embedding 组件
        if embedding is None:
            embedding = EmbeddingFactory.create(settings)
        self._dense_encoder = DenseEncoder(
            embedding=embedding,
            batch_size=None  # BatchProcessor 负责批处理
        )
        self._sparse_encoder = SparseEncoder()
        self._batch_processor = BatchProcessor(
            dense_encoder=self._dense_encoder,
            sparse_encoder=self._sparse_encoder,
            batch_size=self._ingestion_config.batch_size
        )
        
        # Storage 组件（统一存储：chunk+vec+sparse 由 VectorStore 完成，不再使用独立 BM25Indexer）
        if vector_store is None:
            vector_store = VectorStoreFactory.create(settings)
        self._vector_upserter = VectorUpserter(vector_store=vector_store)

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
            if _trace:
                with _trace.stage("integrity_check"):
                    file_hash = self._integrity_checker.compute_sha256(file_path)
                    should_skip = False if force else self._integrity_checker.should_skip(file_hash)
            else:
                file_hash = self._integrity_checker.compute_sha256(file_path)
                should_skip = False if force else self._integrity_checker.should_skip(file_hash)
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
        
        # 步骤 3: Split（切分文档）
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
        
        # 统一存储：图片在 store 阶段经 VectorStore 写入 images 表，不再写入文件

        # 步骤 4: Transform（转换和增强）
        try:
            if _trace:
                with _trace.stage("transform"):
                    transformed_chunks = self._apply_transforms(chunks, trace=trace)
            else:
                transformed_chunks = self._apply_transforms(chunks, trace=trace)
        except Exception as e:
            raise RuntimeError(f"Chunk 转换失败: {str(e)}") from e
        
        # 步骤 5: Encode（编码）
        try:
            if _trace:
                with _trace.stage("encode"):
                    dense_vectors, sparse_vectors = self._batch_processor.process(
                        transformed_chunks, trace=trace
                    )
            else:
                dense_vectors, sparse_vectors = self._batch_processor.process(
                    transformed_chunks, trace=trace
                )
        except Exception as e:
            raise RuntimeError(f"向量编码失败: {str(e)}") from e
        
        # 步骤 6: Store（存储）
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
        
        # 标记处理成功（失败不影响主流程，但需记录日志避免重复处理未被发现）
        try:
            self._integrity_checker.mark_success(file_hash)
        except Exception as e:
            logger.warning("标记文件处理成功失败 (file_hash=%s)，可能导致后续重复处理: %s", file_hash[:16] if file_hash else "?", e)
        
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

        # 统一存储：图片在 store 阶段经 VectorStore 写入 images 表，不再写入文件

        # 步骤 3: Transform
        try:
            if _trace:
                with _trace.stage("transform"):
                    transformed_chunks = self._apply_transforms(chunks, trace=trace)
            else:
                transformed_chunks = self._apply_transforms(chunks, trace=trace)
        except Exception as e:
            raise RuntimeError(f"Chunk 转换失败: {str(e)}") from e

        # 步骤 4: Encode
        try:
            if _trace:
                with _trace.stage("encode"):
                    dense_vectors, sparse_vectors = self._batch_processor.process(
                        transformed_chunks, trace=trace
                    )
            else:
                dense_vectors, sparse_vectors = self._batch_processor.process(
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
        transformed_chunks = []
        
        for chunk in chunks:
            # 依次应用 Transform
            # 1. ChunkRefiner（文本清理和优化）
            chunk = self._chunk_refiner.transform(chunk, trace=trace)
            
            # 2. MetadataEnricher（元数据增强）
            chunk = self._metadata_enricher.transform(chunk, trace=trace)
            
            # 3. ImageCaptioner（图片描述生成）
            chunk = self._image_captioner.transform(chunk, trace=trace)
            
            transformed_chunks.append(chunk)
        
        return transformed_chunks
    
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
        chunk_ids = [c.id for c in chunks]

        self._vector_upserter.upsert_chunks(
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            trace=trace,
            collection_name=collection_name,
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
        if not document or not document.text:
            raise ValueError("Document 或 Document.text 不能为空")
        
        # 使用 Splitter 切分文本
        text_chunks = self._splitter.split_text(document.text, trace=trace)
        
        # 准备阶段：建立图片元数据索引（用于高效查找）
        image_data_dict = document.metadata.get("image_data") or {}
        images_list = document.metadata.get("images") or []
        image_metadata_index: Dict[str, Dict[str, Any]] = {}
        for img_meta in images_list:
            img_id = img_meta.get("image_id")
            if img_id:
                image_metadata_index[img_id] = img_meta
        
        # 将文本片段转换为 Chunk 对象
        chunks: List[Chunk] = []
        current_offset = 0
        
        for idx, chunk_text in enumerate(text_chunks):
            # 生成 Chunk ID（基于文档 ID 和 chunk 索引）
            chunk_id = self._generate_chunk_id(document.id, idx)
            
            # 计算位置偏移量
            start_offset = current_offset
            end_offset = current_offset + len(chunk_text)
            
            # 构建 Chunk 元数据（继承 Document 的元数据，排除图片数据，只传递 chunk 相关的）
            chunk_metadata = {
                k: v for k, v in document.metadata.items()
                if k not in ("image_data", "images")
            }
            chunk_metadata.update({
                "chunk_index": idx,
                "source_doc_id": document.id,
                "total_chunks": len(text_chunks),
            })
            
            # 提取该 chunk 的图片引用
            image_refs = self._extract_image_refs(chunk_text)
            
            # 只传递该 chunk 相关的图片数据和元数据
            if image_refs:
                chunk_metadata["image_refs"] = image_refs
                chunk_image_data: Dict[str, bytes] = {}
                chunk_image_metadata: List[Dict[str, Any]] = []
                for img_id in image_refs:
                    if img_id in image_data_dict:
                        chunk_image_data[img_id] = image_data_dict[img_id]
                    if img_id in image_metadata_index:
                        chunk_image_metadata.append(image_metadata_index[img_id])
                if chunk_image_data:
                    chunk_metadata["image_data"] = chunk_image_data
                if chunk_image_metadata:
                    chunk_metadata["image_metadata"] = chunk_image_metadata
            
            # 创建 Chunk 对象
            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata=chunk_metadata,
                start_offset=start_offset,
                end_offset=end_offset
            )
            
            chunks.append(chunk)
            current_offset = end_offset
        
        return chunks
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """
        生成 Chunk 唯一标识符
        
        Args:
            doc_id: 文档 ID
            chunk_index: Chunk 索引
        
        Returns:
            str: Chunk ID
        """
        # 格式：{doc_id}_chunk_{index}
        return f"{doc_id}_chunk_{chunk_index}"
    
    def _extract_image_refs(self, text: str) -> List[str]:
        """
        从文本中提取图片引用（占位符）
        
        提取格式为 [IMAGE: {image_id}] 的占位符。
        
        Args:
            text: 文本内容
        
        Returns:
            List[str]: 图片 ID 列表
        """
        import re
        # 匹配 [IMAGE: {image_id}] 格式
        pattern = r'\[IMAGE:\s*([^\]]+)\]'
        matches = re.findall(pattern, text)
        return matches
    
    def get_splitter_strategy(self) -> str:
        """
        获取当前使用的 Splitter 策略
        
        Returns:
            str: 策略名称
        """
        return self._splitter.get_strategy()
    
    def get_chunk_size(self) -> int:
        """
        获取当前配置的块大小
        
        Returns:
            int: 块大小
        """
        return self._splitter.get_chunk_size()
    
    def get_chunk_overlap(self) -> int:
        """
        获取当前配置的块重叠大小
        
        Returns:
            int: 重叠大小
        """
        return self._splitter.get_chunk_overlap()
