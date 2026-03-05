"""
SQLite VectorStore 实现

使用 SQLite + sqlite-vec 作为向量存储后端。
Phase 1：chunks + chunks_vec 同库存储。
Phase 3：可选单事务写入 chunks + chunks_vec + chunks_fts（write_fts=True）。
Phase 4：write_fts 时同时写入 images 表，删除时引用计数清理孤立图片。
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
)
from src.core.settings import VectorStoreConfig

logger = logging.getLogger(__name__)

try:
    import sqlite3
    import sqlite_vec
    from sqlite_vec import serialize_float32
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

# metadata 中不序列化到 JSON 的字段
_METADATA_EXCLUDE = ("image_data",)

_FTS5_TABLE = "chunks_fts"
_IMAGES_TABLE = "images"


def _tokenized_text_for_fts(text: str) -> str:
    """对文本分词，用于 FTS5 索引。"""
    from src.libs.tokenizer import tokenize_for_search
    return " ".join(tokenize_for_search(text))


def _prepare_metadata(metadata: Dict[str, Any]) -> str:
    """将 metadata 转为 JSON 字符串，排除不可序列化字段。"""
    safe = {k: v for k, v in metadata.items() if k not in _METADATA_EXCLUDE and not isinstance(v, bytes)}
    return json.dumps(safe, ensure_ascii=False)


class SQLiteVectorStore(BaseVectorStore):
    """
    SQLite VectorStore 实现

    使用 chunks + chunks_vec(vec0) 存储。chunk_rowid 对应 chunks.rowid。
    write_fts=True 时，单事务内同时写入 chunks_fts（Phase 3 统一存储）。
    """

    def __init__(
        self,
        config: VectorStoreConfig,
        write_fts: bool = False,
    ) -> None:
        if not SQLITE_VEC_AVAILABLE:
            raise RuntimeError(
                "sqlite-vec 未安装。请运行: uv add sqlite-vec"
            )
        if not config.sqlite_path:
            raise ValueError("sqlite_path 不能为空")
        if not config.embedding_dim or config.embedding_dim <= 0:
            raise ValueError("embedding_dim 必须大于 0")
        self._config = config
        self._backend = "sqlite"
        self._collection_name = config.collection_name
        self._dim = config.embedding_dim
        self._path = Path(config.sqlite_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._write_fts = bool(write_fts)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path))
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._conn.row_factory = sqlite3.Row
            self._ensure_tables()
        return self._conn

    def _ensure_tables(self) -> None:
        conn = self._conn
        if conn is None:
            return
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                collection_name TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT,
                created_at INTEGER DEFAULT (strftime('%s','now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection_name)")

        # vec0: PK, vector, metadata。collection_name 用于 WHERE 过滤
        dim = self._dim
        try:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                    chunk_rowid INTEGER PRIMARY KEY,
                    embedding float[{dim}] distance_metric=cosine,
                    collection_name TEXT
                )
            """)
        except sqlite3.OperationalError as e:
            if "already exists" in str(e).lower():
                pass
            else:
                raise
        if self._write_fts:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS5_TABLE} USING fts5(
                    chunk_id UNINDEXED,
                    collection_name UNINDEXED,
                    tokenized_text
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {_IMAGES_TABLE} (
                    id TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    image_data BLOB NOT NULL,
                    mime_type TEXT DEFAULT 'image/png',
                    metadata_json TEXT,
                    created_at INTEGER DEFAULT (strftime('%s','now')),
                    PRIMARY KEY (id, collection_name)
                )
            """)
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_images_collection ON {_IMAGES_TABLE}(collection_name)")
        conn.commit()

    def upsert(
        self,
        records: List[VectorRecord],
        trace: Optional[Any] = None,
        collection_name: Optional[str] = None,
        chunks_for_images: Optional[List[Any]] = None,
    ) -> None:
        if not records:
            raise ValueError("记录列表不能为空")
        coll = collection_name or self._collection_name
        if not coll:
            raise ValueError("collection_name 不能为空")

        conn = self._get_conn()
        vec_blob = serialize_float32(records[0].vector)
        if len(vec_blob) // 4 != self._dim:
            raise ValueError(
                f"向量维度与配置不一致: 期望 {self._dim}, 实际 {len(vec_blob) // 4}"
            )

        for i, record in enumerate(records):
            if not isinstance(record, VectorRecord):
                raise ValueError(f"记录 {i} 必须是 VectorRecord 类型")
            if not record.id:
                raise ValueError(f"记录 {i} 的 id 不能为空")
            if not record.vector:
                raise ValueError(f"记录 {i} 的 vector 不能为空")
            if not record.text:
                raise ValueError(f"记录 {i} 的 text 不能为空")
            if len(record.vector) != self._dim:
                raise ValueError(
                    f"记录 {i} 向量维度 {len(record.vector)} 与配置 {self._dim} 不一致"
                )

        try:
            for record in records:
                conn.execute(
                    "DELETE FROM chunks_vec WHERE chunk_rowid IN (SELECT rowid FROM chunks WHERE id = ? AND collection_name = ?)",
                    (record.id, coll),
                )
                if self._write_fts:
                    conn.execute(
                        f"DELETE FROM {_FTS5_TABLE} WHERE chunk_id = ? AND collection_name = ?",
                        (record.id, coll),
                    )
                conn.execute(
                    "DELETE FROM chunks WHERE id = ? AND collection_name = ?",
                    (record.id, coll),
                )
                conn.execute(
                    "INSERT INTO chunks (id, collection_name, text, metadata_json) VALUES (?, ?, ?, ?)",
                    (record.id, coll, record.text, _prepare_metadata(record.metadata)),
                )
                rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                vec_blob = serialize_float32(record.vector)
                conn.execute(
                    "INSERT INTO chunks_vec (chunk_rowid, collection_name, embedding) VALUES (?, ?, ?)",
                    (rowid, coll, vec_blob),
                )
                if self._write_fts:
                    tt = _tokenized_text_for_fts(record.text)
                    conn.execute(
                        f"INSERT INTO {_FTS5_TABLE} (chunk_id, collection_name, tokenized_text) VALUES (?, ?, ?)",
                        (record.id, coll, tt),
                    )
            if self._write_fts and chunks_for_images:
                self._upsert_images(conn, chunks_for_images, coll)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(
                f"SQLite upsert 失败 (backend={self._backend}, collection={coll}): {str(e)}"
            ) from e

    def _upsert_images(
        self,
        conn: sqlite3.Connection,
        chunks: List[Any],
        collection_name: str,
    ) -> None:
        """从 chunks 收集图片并 INSERT INTO images（同事务）。"""
        all_image_ids: set = set()
        image_data_map: Dict[str, bytes] = {}
        image_meta_map: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            refs = getattr(chunk, "metadata", {}).get("image_refs") or []
            for img_id in refs:
                all_image_ids.add(img_id)
                chunk_image_data = getattr(chunk, "metadata", {}).get("image_data") or {}
                if img_id in chunk_image_data and img_id not in image_data_map:
                    image_data_map[img_id] = chunk_image_data[img_id]
                chunk_image_meta = getattr(chunk, "metadata", {}).get("image_metadata") or []
                for m in chunk_image_meta:
                    if m.get("image_id") == img_id and img_id not in image_meta_map:
                        image_meta_map[img_id] = {k: v for k, v in m.items() if not isinstance(v, bytes)}
                        break
        for img_id in all_image_ids:
            img_bytes = image_data_map.get(img_id)
            if not img_bytes:
                continue
            meta = image_meta_map.get(img_id, {})
            mime = meta.get("mime_type", "image/png")
            meta_safe = {k: v for k, v in meta.items() if k != "mime_type" and not isinstance(v, bytes)}
            meta_json = json.dumps(meta_safe, ensure_ascii=False) if meta_safe else None
            conn.execute(
                f"INSERT OR REPLACE INTO {_IMAGES_TABLE} (id, collection_name, image_data, mime_type, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (img_id, collection_name, img_bytes, mime, meta_json),
            )

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> List[QueryResult]:
        coll = collection_name or self._collection_name
        if not coll:
            raise ValueError("collection_name 不能为空")
        if not vector:
            raise ValueError("查询向量不能为空")
        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")
        if len(vector) != self._dim:
            raise ValueError(
                f"查询向量维度 {len(vector)} 与配置 {self._dim} 不一致"
            )

        conn = self._get_conn()
        vec_blob = serialize_float32(vector)

        try:
            rows = conn.execute(
                """
                SELECT c.id, c.text, c.metadata_json, v.distance
                FROM (
                    SELECT chunk_rowid, distance
                    FROM chunks_vec
                    WHERE embedding MATCH ? AND k = ? AND collection_name = ?
                ) v
                JOIN chunks c ON c.rowid = v.chunk_rowid
                ORDER BY v.distance
                LIMIT ?
                """,
                (vec_blob, top_k, coll, top_k),
            ).fetchall()
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                f"SQLite query 失败 (backend={self._backend}, collection={coll}): {str(e)}"
            ) from e

        results = []
        for row in rows:
            meta = {}
            if row["metadata_json"]:
                try:
                    meta = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    pass
            score = 1.0 - float(row["distance"]) if row["distance"] is not None else 0.0
            results.append(
                QueryResult(
                    id=row["id"],
                    score=score,
                    text=row["text"] or "",
                    metadata=meta,
                )
            )
        return results

    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> int:
        if not ids:
            return 0
        coll = collection_name or self._collection_name
        if not coll:
            raise ValueError("collection_name 不能为空")

        conn = self._get_conn()
        try:
            placeholders = ",".join("?" * len(ids))
            if self._write_fts:
                self._delete_orphan_images(conn, ids, coll)
                conn.execute(
                    f"DELETE FROM {_FTS5_TABLE} WHERE chunk_id IN ({placeholders}) AND collection_name = ?",
                    (*ids, coll),
                )
            conn.execute(
                f"DELETE FROM chunks_vec WHERE chunk_rowid IN (SELECT rowid FROM chunks WHERE id IN ({placeholders}) AND collection_name = ?)",
                (*ids, coll),
            )
            conn.execute(
                f"DELETE FROM chunks WHERE id IN ({placeholders}) AND collection_name = ?",
                (*ids, coll),
            )
            conn.commit()
            return len(ids)
        except Exception as e:
            conn.rollback()
            raise RuntimeError(
                f"SQLite delete 失败 (backend={self._backend}, collection={coll}): {str(e)}"
            ) from e

    def _delete_orphan_images(
        self,
        conn: sqlite3.Connection,
        deleted_chunk_ids: List[str],
        collection_name: str,
    ) -> None:
        """删除仅被已删 chunk 引用的图片。"""
        deleted_refs: set = set()
        rows = conn.execute(
            "SELECT metadata_json FROM chunks WHERE id IN ({}) AND collection_name = ?".format(
                ",".join("?" * len(deleted_chunk_ids))
            ),
            (*deleted_chunk_ids, collection_name),
        ).fetchall()
        for row in rows:
            meta = {}
            if row[0]:
                try:
                    meta = json.loads(row[0])
                except json.JSONDecodeError:
                    pass
            for ref in meta.get("image_refs") or []:
                deleted_refs.add(ref)
        if not deleted_refs:
            return
        remaining = conn.execute(
            "SELECT metadata_json FROM chunks WHERE collection_name = ? AND id NOT IN ({})".format(
                ",".join("?" * len(deleted_chunk_ids))
            ),
            (collection_name, *deleted_chunk_ids),
        ).fetchall()
        still_refd: set = set()
        for row in remaining:
            meta = {}
            if row[0]:
                try:
                    meta = json.loads(row[0])
                except json.JSONDecodeError:
                    pass
            for ref in meta.get("image_refs") or []:
                still_refd.add(ref)
        to_delete = deleted_refs - still_refd
        for img_id in to_delete:
            conn.execute(
                f"DELETE FROM {_IMAGES_TABLE} WHERE id = ? AND collection_name = ?",
                (img_id, collection_name),
            )

    def sparse_query(
        self,
        terms: List[str],
        collection_name: str,
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[QueryResult]:
        """
        稀疏检索（FTS5 BM25）。仅当 write_fts=True 时可用。
        """
        if not self._write_fts:
            raise RuntimeError("sparse_query 需要 write_fts=True")
        if not terms:
            return []
        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")
        coll = collection_name or self._collection_name
        if not coll:
            raise ValueError("collection_name 不能为空")

        from src.ingestion.storage.fts5_bm25_indexer import _escape_fts5_term

        safe_terms = [_escape_fts5_term(t) for t in terms if t and str(t).strip()]
        if not safe_terms:
            return []
        query_str = " OR ".join(f'"{t}"' for t in safe_terms)

        conn = self._get_conn()
        try:
            rows = conn.execute(
                f"""
                SELECT c.id, c.text, c.metadata_json, fts.rk
                FROM (
                    SELECT chunk_id, bm25({_FTS5_TABLE}) AS rk
                    FROM {_FTS5_TABLE}
                    WHERE {_FTS5_TABLE} MATCH ? AND collection_name = ?
                    ORDER BY rk
                    LIMIT ?
                ) fts
                JOIN chunks c ON c.id = fts.chunk_id AND c.collection_name = ?
                """,
                (query_str, coll, top_k, coll),
            ).fetchall()
        except sqlite3.OperationalError as e:
            if "syntax error" in str(e).lower() or "malformed" in str(e).lower():
                return []
            raise

        results = []
        for row in rows:
            meta = {}
            if row["metadata_json"]:
                try:
                    meta = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    pass
            # FTS5 bm25 越小越相关，取负值使分数越大越好
            score = -float(row["rk"] or 0)
            results.append(
                QueryResult(
                    id=row["id"],
                    score=score,
                    text=row["text"] or "",
                    metadata=meta,
                )
            )
        return results

    def sparse_index_exists(self, collection_name: str) -> bool:
        """检查 collection 的 FTS5 索引是否有数据。"""
        if not collection_name:
            return False
        if not self._write_fts:
            return False
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT 1 FROM {_FTS5_TABLE} WHERE collection_name = ? LIMIT 1",
            (collection_name,),
        ).fetchone()
        return row is not None

    def list_collections(self) -> List[str]:
        """列出所有 collection 名称。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT collection_name FROM chunks ORDER BY collection_name"
        ).fetchall()
        return [r[0] for r in rows]

    def get_image(
        self,
        image_id: str,
        collection_name: str,
    ) -> Optional[tuple]:
        """
        从 images 表加载单张图片。

        Args:
            image_id: 图片 ID
            collection_name: 集合名称

        Returns:
            (image_data: bytes, mime_type: str) 或 None
        """
        if not self._write_fts:
            return None
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT image_data, mime_type FROM {_IMAGES_TABLE} WHERE id = ? AND collection_name = ?",
            (image_id, collection_name),
        ).fetchone()
        if row:
            return (row[0], row[1] or "image/png")
        return None

    def get_backend(self) -> str:
        return self._backend

    def get_collection_name(self) -> str:
        return self._collection_name

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as e:
                logger.debug("SQLite 连接关闭时异常: %s", e)
            self._conn = None
