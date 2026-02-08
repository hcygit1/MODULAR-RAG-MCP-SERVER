"""
Ingestion 核心数据模型

定义 ingestion 与 retrieval 共用的数据结构。
"""
from typing import Dict, Any, Optional
import json


class Document:
    """
    Document 数据结构
    
    表示一个完整的文档，包含文本内容和元数据。
    用于 Loader 的输出和 Pipeline 的输入。
    """
    
    def __init__(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 Document
        
        Args:
            id: 文档唯一标识符
            text: 文档文本内容（通常是 Markdown 格式）
            metadata: 元数据字典（可选），包含 source_path, doc_type, title 等
        """
        if not id:
            raise ValueError("Document id 不能为空")
        if not text:
            raise ValueError("Document text 不能为空")
        
        self.id = id
        self.text = text
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 包含 id, text, metadata 的字典
        """
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """
        转换为 JSON 字符串
        
        Returns:
            str: JSON 格式的字符串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        从字典创建 Document 实例
        
        Args:
            data: 包含 id, text, metadata 的字典
        
        Returns:
            Document: Document 实例
        
        Raises:
            ValueError: 当字典缺少必需字段时
        """
        if "id" not in data:
            raise ValueError("字典缺少 'id' 字段")
        if "text" not in data:
            raise ValueError("字典缺少 'text' 字段")
        
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "Document":
        """
        从 JSON 字符串创建 Document 实例
        
        Args:
            json_str: JSON 格式的字符串
        
        Returns:
            Document: Document 实例
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __eq__(self, other: Any) -> bool:
        """判断两个 Document 是否相等"""
        if not isinstance(other, Document):
            return False
        return (
            self.id == other.id and
            self.text == other.text and
            self.metadata == other.metadata
        )
    
    def __repr__(self) -> str:
        """返回 Document 的字符串表示"""
        return f"Document(id={self.id!r}, text_length={len(self.text)}, metadata_keys={list(self.metadata.keys())})"


class Chunk:
    """
    Chunk 数据结构
    
    表示文档的一个片段（chunk），包含文本内容、元数据和位置信息。
    用于 Splitter 的输出和后续 Transform/Embedding 的输入。
    """
    
    def __init__(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None
    ):
        """
        初始化 Chunk
        
        Args:
            id: Chunk 唯一标识符
            text: Chunk 文本内容
            metadata: 元数据字典（可选），包含 source_path, chunk_index, section_path 等
            start_offset: 在原始文档中的起始位置（字符偏移量，可选）
            end_offset: 在原始文档中的结束位置（字符偏移量，可选）
        """
        if not id:
            raise ValueError("Chunk id 不能为空")
        if not text:
            raise ValueError("Chunk text 不能为空")
        
        self.id = id
        self.text = text
        self.metadata = metadata or {}
        self.start_offset = start_offset
        self.end_offset = end_offset
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 包含 id, text, metadata, start_offset, end_offset 的字典
        """
        result = {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata
        }
        
        if self.start_offset is not None:
            result["start_offset"] = self.start_offset
        if self.end_offset is not None:
            result["end_offset"] = self.end_offset
        
        return result
    
    def to_json(self) -> str:
        """
        转换为 JSON 字符串
        
        Returns:
            str: JSON 格式的字符串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """
        从字典创建 Chunk 实例
        
        Args:
            data: 包含 id, text, metadata, start_offset, end_offset 的字典
        
        Returns:
            Chunk: Chunk 实例
        
        Raises:
            ValueError: 当字典缺少必需字段时
        """
        if "id" not in data:
            raise ValueError("字典缺少 'id' 字段")
        if "text" not in data:
            raise ValueError("字典缺少 'text' 字段")
        
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            start_offset=data.get("start_offset"),
            end_offset=data.get("end_offset")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "Chunk":
        """
        从 JSON 字符串创建 Chunk 实例
        
        Args:
            json_str: JSON 格式的字符串
        
        Returns:
            Chunk: Chunk 实例
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __eq__(self, other: Any) -> bool:
        """判断两个 Chunk 是否相等"""
        if not isinstance(other, Chunk):
            return False
        return (
            self.id == other.id and
            self.text == other.text and
            self.metadata == other.metadata and
            self.start_offset == other.start_offset and
            self.end_offset == other.end_offset
        )
    
    def __repr__(self) -> str:
        """返回 Chunk 的字符串表示"""
        offset_info = ""
        if self.start_offset is not None and self.end_offset is not None:
            offset_info = f", offset=[{self.start_offset}:{self.end_offset}]"
        return f"Chunk(id={self.id!r}, text_length={len(self.text)}, metadata_keys={list(self.metadata.keys())}{offset_info})"
