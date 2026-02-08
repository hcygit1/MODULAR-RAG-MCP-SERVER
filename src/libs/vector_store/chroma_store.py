"""
Chroma VectorStore 实现

使用 ChromaDB 作为向量存储后端。
支持本地持久化目录，无需额外数据库服务。
"""
from typing import List, Dict, Any, Optional
import os

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult
)
from src.core.settings import VectorStoreConfig

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaStore(BaseVectorStore):
    """
    Chroma VectorStore 实现
    
    使用 ChromaDB 作为向量存储后端。
    支持本地持久化，数据存储在指定目录中。
    """
    
    def __init__(self, config: VectorStoreConfig):
        """
        初始化 Chroma Store
        
        Args:
            config: VectorStore 配置对象
        """
        if not CHROMADB_AVAILABLE:
            raise RuntimeError(
                "ChromaDB 未安装。请安装: pip install chromadb"
            )
        
        if not config.collection_name:
            raise ValueError("Collection name 不能为空")
        
        self._config = config
        self._backend = "chroma"
        self._collection_name = config.collection_name
        self._persist_path = config.persist_path
        
        # 确保持久化目录存在
        if self._persist_path:
            os.makedirs(self._persist_path, exist_ok=True)
        
        # 创建 ChromaDB 客户端
        # 如果指定了 persist_path，使用持久化模式
        if self._persist_path:
            self._client = chromadb.PersistentClient(
                path=self._persist_path
            )
        else:
            # 内存模式（仅用于测试）
            self._client = chromadb.Client()
        
        # 获取或创建集合
        # ChromaDB 会自动创建集合（如果不存在）
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )
    
    def upsert(
        self,
        records: List[VectorRecord],
        trace: Optional[Any] = None
    ) -> None:
        """
        批量插入或更新向量记录（幂等操作）
        
        Args:
            records: 向量记录列表
            trace: 追踪上下文（可选）
        
        Raises:
            ValueError: 当记录格式不正确时
            RuntimeError: 当存储操作失败时
        """
        if not records:
            raise ValueError("记录列表不能为空")
        
        # 验证记录格式
        for i, record in enumerate(records):
            if not isinstance(record, VectorRecord):
                raise ValueError(f"记录 {i} 必须是 VectorRecord 类型，得到: {type(record)}")
            if not record.id:
                raise ValueError(f"记录 {i} 的 id 不能为空")
            if not record.vector:
                raise ValueError(f"记录 {i} 的 vector 不能为空")
            if not isinstance(record.vector, list):
                raise ValueError(f"记录 {i} 的 vector 必须是列表类型")
            if not record.text:
                raise ValueError(f"记录 {i} 的 text 不能为空")
        
        try:
            # 准备数据
            ids = [record.id for record in records]
            embeddings = [record.vector for record in records]
            documents = [record.text for record in records]
            metadatas = [record.metadata for record in records]
            
            # ChromaDB upsert（如果 id 存在则更新，否则插入）
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            raise RuntimeError(
                f"ChromaDB upsert 失败 (backend={self._backend}, collection={self._collection_name}): {str(e)}"
            ) from e
    
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None
    ) -> List[QueryResult]:
        """
        向量相似度查询
        
        Args:
            vector: 查询向量
            top_k: 返回最相似的 top_k 条记录
            filters: 元数据过滤条件（可选）
            trace: 追踪上下文（可选）
        
        Returns:
            List[QueryResult]: 查询结果列表，按相似度分数降序排列
        
        Raises:
            ValueError: 当向量维度不匹配或 top_k <= 0 时
            RuntimeError: 当查询操作失败时
        """
        if not vector:
            raise ValueError("查询向量不能为空")
        
        if not isinstance(vector, list):
            raise ValueError("查询向量必须是列表类型")
        
        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")
        
        try:
            # 构建查询
            # ChromaDB 的 where 过滤器格式：{"metadata_key": "value"} 或 {"metadata_key": {"$in": ["value1", "value2"]}}
            where = None
            if filters:
                where = filters
            
            # 执行查询
            results = self._collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where
            )
            
            # 转换结果为 QueryResult 列表
            query_results = []
            
            # ChromaDB 返回格式：
            # {
            #     "ids": [[id1, id2, ...]],
            #     "distances": [[dist1, dist2, ...]],  # 距离（越小越相似）
            #     "documents": [[doc1, doc2, ...]],
            #     "metadatas": [[meta1, meta2, ...]]
            # }
            if results["ids"] and len(results["ids"]) > 0:
                ids = results["ids"][0]
                distances = results["distances"][0] if results.get("distances") else []
                documents = results["documents"][0] if results.get("documents") else []
                metadatas = results["metadatas"][0] if results.get("metadatas") else []
                
                for i, record_id in enumerate(ids):
                    # ChromaDB 返回的是距离（distance），需要转换为相似度分数（score）
                    # 距离越小，相似度越高
                    # 使用 1 / (1 + distance) 转换为相似度分数，范围 [0, 1]
                    distance = distances[i] if i < len(distances) else 0.0
                    score = 1.0 / (1.0 + distance) if distance >= 0 else 1.0
                    
                    document = documents[i] if i < len(documents) else ""
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    query_results.append(
                        QueryResult(
                            id=record_id,
                            score=score,
                            text=document,
                            metadata=metadata
                        )
                    )
            
            return query_results
        except Exception as e:
            raise RuntimeError(
                f"ChromaDB query 失败 (backend={self._backend}, collection={self._collection_name}): {str(e)}"
            ) from e
    
    def get_backend(self) -> str:
        """获取后端名称"""
        return self._backend
    
    def get_collection_name(self) -> str:
        """获取集合名称"""
        return self._collection_name
