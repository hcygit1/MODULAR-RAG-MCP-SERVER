"""
Ingestion Pipeline 模块

提供文档摄取、处理、存储的完整流程。
实现 integrity→load→split→transform→encode→store 的完整编排。
"""
from typing import List, Optional, Any, Dict
from pathlib import Path

from src.ingestion.models import Document, Chunk
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.loader.file_integrity import FileIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.core.settings import Settings

from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage


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
        
        # Transform 组件
        self._chunk_refiner = ChunkRefiner(
            config=self._ingestion_config,
            llm=llm
        )
        self._metadata_enricher = MetadataEnricher(
            config=self._ingestion_config,
            llm=llm
        )
        self._image_captioner = ImageCaptioner(
            config=self._ingestion_config,
            vision_llm_config=settings.vision_llm,
            vision_llm=vision_llm
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
        
        # Storage 组件
        if vector_store is None:
            vector_store = VectorStoreFactory.create(settings)
        self._vector_upserter = VectorUpserter(vector_store=vector_store)
        self._bm25_indexer = BM25Indexer()
        self._image_storage = ImageStorage()
    
    def process(
        self,
        file_path: str,
        collection_name: str,
        trace: Optional[Any] = None
    ) -> None:
        """
        处理单个文件，执行完整的 ingestion 流程
        
        Args:
            file_path: 文件路径
            collection_name: 集合名称，用于组织存储
            trace: 追踪上下文（可选）
        
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
        
        # 步骤 1: Integrity Check（文件完整性检查）
        try:
            file_hash = self._integrity_checker.compute_sha256(file_path)
            if self._integrity_checker.should_skip(file_hash):
                # 文件未变更，跳过处理
                return
        except Exception as e:
            raise RuntimeError(f"文件完整性检查失败: {str(e)}") from e
        
        # 步骤 2: Load（加载文档）
        try:
            loader = PdfLoader()  # 当前只支持 PDF
            document = loader.load(file_path, trace=trace)
        except Exception as e:
            raise RuntimeError(f"文档加载失败: {str(e)}") from e
        
        # 步骤 3: Split（切分文档）
        try:
            chunks = self.split_document(document, trace=trace)
            if not chunks:
                raise RuntimeError("文档切分后未产生任何 chunks")
        except Exception as e:
            raise RuntimeError(f"文档切分失败: {str(e)}") from e
        
        # 步骤 4: Transform（转换和增强）
        try:
            transformed_chunks = self._apply_transforms(chunks, trace=trace)
        except Exception as e:
            raise RuntimeError(f"Chunk 转换失败: {str(e)}") from e
        
        # 步骤 5: Encode（编码）
        try:
            dense_vectors, sparse_vectors = self._batch_processor.process(
                transformed_chunks,
                trace=trace
            )
        except Exception as e:
            raise RuntimeError(f"向量编码失败: {str(e)}") from e
        
        # 步骤 6: Store（存储）
        try:
            self._store_results(
                transformed_chunks,
                dense_vectors,
                sparse_vectors,
                collection_name,
                trace=trace
            )
        except Exception as e:
            raise RuntimeError(f"存储失败: {str(e)}") from e
        
        # 标记处理成功
        try:
            self._integrity_checker.mark_success(file_hash)
        except Exception as e:
            # 标记失败不影响主流程，但记录警告
            # TODO: 记录到日志
            pass
    
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
        存储处理结果
        
        Args:
            chunks: Chunk 列表
            dense_vectors: 稠密向量列表
            sparse_vectors: 稀疏向量列表
            collection_name: 集合名称
            trace: 追踪上下文（可选）
        """
        # 1. 存储向量到向量数据库
        self._vector_upserter.upsert_chunks(
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            trace=trace
        )
        
        # 2. 构建并保存 BM25 索引
        self._bm25_indexer.build(
            chunks=chunks,
            sparse_vectors=sparse_vectors,
            collection_name=collection_name,
            trace=trace
        )
        self._bm25_indexer.save()
        
        # 3. 保存图片（如果有）
        # 从 chunks 中提取所有图片引用并保存
        self._save_images(chunks, collection_name, trace=trace)
        
        # 保存图片索引
        self._image_storage.save_index(collection_name)
    
    def _save_images(
        self,
        chunks: List[Chunk],
        collection_name: str,
        trace: Optional[Any] = None
    ) -> None:
        """
        保存 chunks 中引用的图片
        
        Args:
            chunks: Chunk 列表
            collection_name: 集合名称
            trace: 追踪上下文（可选）
        
        注意：当前实现假设图片数据已经在 Document.metadata 中。
        实际实现中，图片数据应该从 Loader 阶段获取。
        """
        # 收集所有图片引用
        all_image_refs = set()
        for chunk in chunks:
            image_refs = chunk.metadata.get("image_refs", [])
            all_image_refs.update(image_refs)
        
        # 如果有图片引用，尝试从 metadata 中获取图片数据并保存
        # 注意：当前实现简化处理，实际应该从 Loader 阶段获取图片二进制数据
        # 这里只是占位实现，确保流程完整
        for image_id in all_image_refs:
            # TODO: 从 Document.metadata 或图片缓存中获取图片数据
            # 当前简化：如果图片数据不存在，跳过
            # 实际实现应该在 Loader 阶段提取图片并暂存
            pass
    
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
        
        # 将文本片段转换为 Chunk 对象
        chunks: List[Chunk] = []
        current_offset = 0
        
        for idx, chunk_text in enumerate(text_chunks):
            # 生成 Chunk ID（基于文档 ID 和 chunk 索引）
            chunk_id = self._generate_chunk_id(document.id, idx)
            
            # 计算位置偏移量
            start_offset = current_offset
            end_offset = current_offset + len(chunk_text)
            
            # 构建 Chunk 元数据（继承 Document 的元数据，并添加 chunk 特定信息）
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunk_index": idx,
                "source_doc_id": document.id,
                "total_chunks": len(text_chunks),
            })
            
            # 如果 Document 有图片引用，尝试保持关联
            if "images" in document.metadata:
                # 检查 chunk 文本中是否包含图片占位符
                # 如果包含，提取对应的 image_id
                image_refs = self._extract_image_refs(chunk_text)
                if image_refs:
                    chunk_metadata["image_refs"] = image_refs
            
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
