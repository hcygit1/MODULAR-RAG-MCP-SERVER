"""
Ingestion Pipeline 模块

提供文档摄取、处理、存储的完整流程。
当前阶段实现 Loader -> Splitter 的集成。
"""
from typing import List, Optional, Any

from src.ingestion.models import Document, Chunk
from src.libs.splitter.splitter_factory import SplitterFactory
from src.core.settings import Settings


class IngestionPipeline:
    """
    Ingestion Pipeline 类
    
    负责将文档通过 Loader -> Splitter -> Transform -> Embedding -> Storage 流程处理。
    当前阶段（C4）实现 Loader -> Splitter 的集成。
    """
    
    def __init__(self, settings: Settings, splitter_strategy: Optional[str] = None):
        """
        初始化 Ingestion Pipeline
        
        Args:
            settings: 配置对象，包含所有组件的配置
            splitter_strategy: Splitter 策略（可选），如果不提供则使用默认策略
        """
        self._settings = settings
        self._splitter = SplitterFactory.create(settings, strategy=splitter_strategy)
    
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
            # 注意：这里简化处理，后续 C7 阶段会完善图片关联逻辑
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
