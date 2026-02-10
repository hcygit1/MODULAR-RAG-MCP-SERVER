"""
BM25 Indexer 实现

将 sparse encoder 输出构建为倒排索引并持久化到文件系统。
支持索引构建、加载和查询功能。
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

from src.ingestion.models import Chunk


class BM25Indexer:
    """
    BM25 Indexer 实现
    
    负责构建、存储和查询 BM25 倒排索引。
    索引结构：term -> [(chunk_id, weight), ...]
    """
    
    def __init__(self, base_path: str = "data/db/bm25"):
        """
        初始化 BM25Indexer
        
        Args:
            base_path: 索引文件存储的基础路径，默认为 "data/db/bm25"
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        
        # 内存中的索引结构
        self._inverted_index: Dict[str, List[Tuple[str, float]]] = {}  # term -> [(chunk_id, weight), ...]
        self._chunk_metadata: Dict[str, Dict[str, Any]] = {}  # chunk_id -> metadata
        self._collection_name: Optional[str] = None
    
    def build(
        self,
        chunks: List[Chunk],
        sparse_vectors: List[Dict[str, float]],
        collection_name: str,
        trace: Optional[Any] = None
    ) -> None:
        """
        构建倒排索引
        
        Args:
            chunks: Chunk 对象列表
            sparse_vectors: 稀疏向量列表（Term Weights），每个 Chunk 对应一个字典
            collection_name: 集合名称，用于组织索引文件
            trace: 追踪上下文（可选）
        
        Raises:
            ValueError: 当输入参数不匹配时
        """
        if not chunks:
            raise ValueError("chunks 列表不能为空")
        
        if len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"chunks 数量 ({len(chunks)}) 与 sparse_vectors 数量 ({len(sparse_vectors)}) 不一致"
            )
        
        if not collection_name:
            raise ValueError("collection_name 不能为空")
        
        self._collection_name = collection_name
        
        # 清空现有索引
        self._inverted_index = defaultdict(list)
        self._chunk_metadata = {}
        
        # 构建倒排索引
        for chunk, sparse_vector in zip(chunks, sparse_vectors):
            chunk_id = chunk.id
            
            # 保存 chunk metadata（排除不可 JSON 序列化的字段如 image_data）
            metadata_safe = {
                k: v for k, v in chunk.metadata.items()
                if k not in ("image_data",) and not isinstance(v, bytes)
            }
            
            self._chunk_metadata[chunk_id] = {
                "text": chunk.text,
                "metadata": metadata_safe,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset
            }
            
            # 将 term weights 添加到倒排索引
            for term, weight in sparse_vector.items():
                if weight != 0:  # 只存储非零权重
                    self._inverted_index[term].append((chunk_id, weight))
        
        # 对每个 term 的 posting list 按权重降序排序（便于查询时快速取 top）
        for term in self._inverted_index:
            self._inverted_index[term].sort(key=lambda x: x[1], reverse=True)
    
    def save(self) -> None:
        """
        将索引保存到文件系统
        
        文件结构：
        - {base_path}/{collection_name}/index.json: 倒排索引和元数据
        """
        if not self._collection_name:
            raise RuntimeError("索引未构建，无法保存。请先调用 build()")
        
        collection_path = self._base_path / self._collection_name
        collection_path.mkdir(parents=True, exist_ok=True)
        
        index_file = collection_path / "index.json"
        
        # 准备保存的数据
        index_data = {
            "collection_name": self._collection_name,
            "inverted_index": {
                term: postings for term, postings in self._inverted_index.items()
            },
            "chunk_metadata": self._chunk_metadata,
            "total_chunks": len(self._chunk_metadata),
            "total_terms": len(self._inverted_index)
        }
        
        # 保存为 JSON
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    def load(self, collection_name: str) -> None:
        """
        从文件系统加载索引
        
        Args:
            collection_name: 集合名称
        
        Raises:
            FileNotFoundError: 当索引文件不存在时
            ValueError: 当索引文件格式不正确时
        """
        if not collection_name:
            raise ValueError("collection_name 不能为空")
        
        self._collection_name = collection_name
        collection_path = self._base_path / collection_name
        index_file = collection_path / "index.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_file}")
        
        # 加载 JSON
        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        
        # 验证数据格式
        if "inverted_index" not in index_data or "chunk_metadata" not in index_data:
            raise ValueError(f"索引文件格式不正确: {index_file}")
        
        # 恢复索引结构
        self._inverted_index = {}
        for term, postings in index_data["inverted_index"].items():
            # postings 是列表，每个元素是 [chunk_id, weight]
            self._inverted_index[term] = [(pid, weight) for pid, weight in postings]
        
        self._chunk_metadata = index_data["chunk_metadata"]
    
    def query(
        self,
        terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None
    ) -> List[Tuple[str, float]]:
        """
        查询索引，返回 top-k chunk_ids 及其分数
        
        Args:
            terms: 查询词列表
            top_k: 返回最相关的 top_k 个结果
        
        Returns:
            List[Tuple[str, float]]: (chunk_id, score) 列表，按分数降序排列
                                   - 分数是查询词权重的累加和
                                   - 结果数量 <= top_k
        
        Raises:
            RuntimeError: 当索引未加载时
        """
        if not self._inverted_index:
            raise RuntimeError("索引未加载，无法查询。请先调用 load() 或 build()")
        
        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")
        
        # 累加每个查询词的权重
        chunk_scores: Dict[str, float] = defaultdict(float)
        
        for term in terms:
            term_lower = term.lower()  # 转换为小写以匹配索引
            
            if term_lower in self._inverted_index:
                # 累加该 term 在所有文档中的权重
                for chunk_id, weight in self._inverted_index[term_lower]:
                    chunk_scores[chunk_id] += weight
        
        # 按分数降序排序，返回 top_k
        sorted_results = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def get_collection_name(self) -> Optional[str]:
        """
        获取当前集合名称
        
        Returns:
            Optional[str]: 集合名称，如果未设置则返回 None
        """
        return self._collection_name
    
    def get_index_path(self) -> Optional[Path]:
        """
        获取索引文件路径
        
        Returns:
            Optional[Path]: 索引文件路径，如果未设置集合名称则返回 None
        """
        if not self._collection_name:
            return None
        return self._base_path / self._collection_name / "index.json"
    
    def get_total_chunks(self) -> int:
        """
        获取索引中的 chunk 总数
        
        Returns:
            int: chunk 总数
        """
        return len(self._chunk_metadata)
    
    def get_total_terms(self) -> int:
        """
        获取索引中的 term 总数
        
        Returns:
            int: term 总数
        """
        return len(self._inverted_index)
