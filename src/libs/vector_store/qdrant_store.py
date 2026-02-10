"""
Qdrant VectorStore 实现

使用 Qdrant 作为向量存储后端。
支持 list、dict 等复杂 metadata 类型，无需序列化。
"""
import uuid
from typing import List, Dict, Any, Optional

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult
)
from src.core.settings import VectorStoreConfig

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        MatchAny,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


def _to_qdrant_id(record_id: str) -> str:
    """
    将 record_id（如 SHA256 hex）转为 Qdrant 可接受的 UUID 格式。
    Qdrant 要求 id 为 int 或 UUID 字符串，此处取前 32 位 hex 构造 UUID。
    """
    if not record_id or len(record_id) < 32:
        return str(uuid.uuid4())
    return str(uuid.UUID(hex=record_id[:32]))


def _prepare_payload(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备 Qdrant payload，排除不可序列化的字段。
    
    Qdrant 支持 list、dict 等 JSON 类型，但需排除 bytes。
    image_data 包含图片二进制，需排除（图片由 ImageStorage 单独存储）。
    """
    EXCLUDE_KEYS = ("image_data",)
    result = {}
    for k, v in metadata.items():
        if k in EXCLUDE_KEYS or isinstance(v, bytes):
            continue
        result[k] = v
    return result


class QdrantStore(BaseVectorStore):
    """
    Qdrant VectorStore 实现
    
    使用 Qdrant 作为向量存储后端。
    支持 tags、image_refs 等 list/dict 类型的 metadata，无需序列化。
    """
    
    def __init__(self, config: VectorStoreConfig):
        """
        初始化 Qdrant Store
        
        Args:
            config: VectorStore 配置对象
        """
        if not QDRANT_AVAILABLE:
            raise RuntimeError(
                "qdrant-client 未安装。请安装: pip install qdrant-client"
            )
        
        if not config.collection_name:
            raise ValueError("collection_name 不能为空")
        
        self._config = config
        self._backend = "qdrant"
        self._collection_name = config.collection_name
        
        # 创建 Qdrant 客户端
        # qdrant_url 为空时使用 persist_path 做 embedded；否则连接远程
        if config.persist_path and not config.qdrant_url:
            # 本地 embedded 模式
            self._client = QdrantClient(path=config.persist_path)
        else:
            # 远程模式（默认 http://localhost:6333）
            url = config.qdrant_url or "http://localhost:6333"
            self._client = QdrantClient(
                url=url,
                api_key=config.qdrant_api_key if config.qdrant_api_key else None,
            )
        
        # 集合在首次 upsert 时按需创建
        self._collection_initialized = False
    
    def _ensure_collection(self, vector_size: int) -> None:
        """按需创建集合（若不存在）"""
        if self._collection_initialized:
            return
        try:
            self._client.get_collection(self._collection_name)
            self._collection_initialized = True
            return
        except Exception:
            pass
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        self._collection_initialized = True
    
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
            vector_size = len(records[0].vector)
            self._ensure_collection(vector_size)
            
            points = []
            for record in records:
                payload = _prepare_payload(record.metadata)
                payload["text"] = record.text
                payload["_original_id"] = record.id  # 保留原始 id 便于检索时还原
                
                points.append(
                    PointStruct(
                        id=_to_qdrant_id(record.id),
                        vector=record.vector,
                        payload=payload,
                    )
                )
            
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Qdrant upsert 失败 (backend={self._backend}, collection={self._collection_name}): {str(e)}"
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
            query_filter = None
            if filters and len(filters) > 0:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, dict) and "$in" in value:
                        conditions.append(
                            FieldCondition(key=key, match=MatchAny(any=value["$in"]))
                        )
                    else:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                if conditions:
                    query_filter = Filter(must=conditions)
            
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            
            query_results = []
            for hit in response.points or []:
                payload = dict(hit.payload or {})
                text = payload.pop("text", "")
                original_id = payload.pop("_original_id", str(hit.id))
                query_results.append(
                    QueryResult(
                        id=original_id,
                        score=float(hit.score),
                        text=text,
                        metadata=payload
                    )
                )
            return query_results
        except Exception as e:
            raise RuntimeError(
                f"Qdrant query 失败 (backend={self._backend}, collection={self._collection_name}): {str(e)}"
            ) from e
    
    def close(self) -> None:
        """关闭 Qdrant 客户端，避免进程退出时 __del__ 报错"""
        if hasattr(self, "_client") and self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def get_backend(self) -> str:
        """获取后端名称"""
        return self._backend

    def get_collection_name(self) -> str:
        """获取集合名称"""
        return self._collection_name
