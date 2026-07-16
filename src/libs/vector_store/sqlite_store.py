"""SQLite + sqlite-vec VectorStore Facade。"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from src.core.settings import VectorStoreConfig
from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    QueryResult,
    VectorRecord,
)
from src.libs.vector_store.sqlite_image_repository import SQLiteImageRepository
from src.libs.vector_store.sqlite_record_repository import (
    SQLiteRecordRepository,
    prepare_metadata,
    tokenized_text_for_fts,
)
from src.libs.vector_store.sqlite_schema import SQLiteSchemaManager
from src.libs.vector_store.sqlite_search import SQLiteSearchService

logger = logging.getLogger(__name__)

try:
    import sqlite_vec  # type: ignore[import-untyped]

    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False


def _tokenized_text_for_fts(text: str) -> str:
    """兼容原模块级 helper。"""
    return tokenized_text_for_fts(text)


def _prepare_metadata(metadata: Dict[str, Any]) -> str:
    """兼容原模块级 helper。"""
    return prepare_metadata(metadata)


class SQLiteVectorStore(BaseVectorStore):
    """协调 SQLite 连接、事务和各持久化组件。"""

    def __init__(
        self,
        config: VectorStoreConfig,
        write_fts: bool = False,
    ) -> None:
        if not SQLITE_VEC_AVAILABLE:
            raise RuntimeError("sqlite-vec 未安装。请运行: uv add sqlite-vec")
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
        self._schema = SQLiteSchemaManager(self._dim, self._write_fts)
        self._images = SQLiteImageRepository()
        self._records = SQLiteRecordRepository(
            self._dim,
            self._write_fts,
            self._images,
        )
        self._search = SQLiteSearchService()

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
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._schema.ensure_tables(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _create_chunks_table(
        conn: sqlite3.Connection,
        table_name: str = "chunks",
    ) -> None:
        SQLiteSchemaManager.create_chunks_table(conn, table_name)

    def _ensure_chunks_table(self, conn: sqlite3.Connection) -> None:
        migration_required = self._schema.chunks_table_needs_migration(conn)
        if not migration_required:
            self._schema.ensure_chunks_table(conn)
            return
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._schema.ensure_chunks_table(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def upsert(
        self,
        records: List[VectorRecord],
        trace: Optional[Any] = None,
        collection_name: Optional[str] = None,
        chunks_for_images: Optional[List[Any]] = None,
    ) -> None:
        if not records:
            raise ValueError("记录列表不能为空")
        collection = collection_name or self._collection_name
        if not collection:
            raise ValueError("collection_name 不能为空")

        self._validate_records(records)
        conn = self._get_conn()
        try:
            self._write_records(conn, records, collection, chunks_for_images)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(
                f"SQLite upsert 失败 (backend={self._backend}, "
                f"collection={collection}): {str(exc)}"
            ) from exc

    def _validate_records(self, records: List[VectorRecord]) -> None:
        self._records.validate_records(records)

    def _write_records(
        self,
        conn: sqlite3.Connection,
        records: List[VectorRecord],
        collection_name: str,
        chunks_for_images: Optional[List[Any]] = None,
        *,
        orphan_candidates: Optional[set[str]] = None,
    ) -> None:
        self._records.write_records(
            conn,
            records,
            collection_name,
            chunks_for_images,
            orphan_candidates=orphan_candidates,
            upsert_images=self._upsert_images,
        )

    def replace_document_records(
        self,
        source_doc_id: str,
        records: List[VectorRecord],
        collection_name: str,
        trace: Optional[Any] = None,
        chunks_for_images: Optional[List[Any]] = None,
    ) -> int:
        """在单个 SQLite 事务中删除旧文档并写入新版本。"""
        collection = (collection_name or "").strip()
        if not collection:
            raise ValueError("collection_name 不能为空")
        document_id = (source_doc_id or "").strip()
        if not document_id:
            raise ValueError("source_doc_id 不能为空")
        if not records:
            raise ValueError("记录列表不能为空")
        self._validate_records(records)

        conn = self._get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            old_ids = self._records.find_ids_by_source_doc_id(
                conn,
                collection,
                document_id,
            )
            orphan_candidates = self._images.collect_image_refs(
                conn,
                old_ids,
                collection,
            )
            self._delete_records(
                conn,
                old_ids,
                collection,
                cleanup_images=False,
            )
            self._write_records(
                conn,
                records,
                collection,
                chunks_for_images,
                orphan_candidates=orphan_candidates,
            )
            conn.commit()
            return len(old_ids)
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(
                f"SQLite 文档替换失败 (backend={self._backend}, "
                f"collection={collection}): {str(exc)}"
            ) from exc

    def _upsert_images(
        self,
        conn: sqlite3.Connection,
        chunks: List[Any],
        collection_name: str,
    ) -> None:
        self._images.upsert_images(conn, chunks, collection_name)

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> List[QueryResult]:
        collection = collection_name or self._collection_name
        if not collection:
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
        plan = self._search.prepare_dense_query(
            conn,
            top_k,
            collection,
            filters,
            filter_builder=self._build_filter_sql,
        )
        try:
            return self._search.execute_dense_query(
                conn,
                vector,
                plan,
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"SQLite query 失败 (backend={self._backend}, "
                f"collection={collection}): {str(exc)}"
            ) from exc

    @staticmethod
    def _build_filter_sql(
        filters: Optional[Dict[str, Any]],
    ) -> tuple[str, List[Any]]:
        return SQLiteSearchService.build_filter_sql(filters)

    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> int:
        if not ids:
            return 0
        collection = collection_name or self._collection_name
        if not collection:
            raise ValueError("collection_name 不能为空")

        conn = self._get_conn()
        try:
            deleted = self._delete_records(conn, ids, collection)
            conn.commit()
            return deleted
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(
                f"SQLite delete 失败 (backend={self._backend}, "
                f"collection={collection}): {str(exc)}"
            ) from exc

    def _delete_records(
        self,
        conn: sqlite3.Connection,
        ids: List[str],
        collection_name: str,
        *,
        cleanup_images: bool = True,
    ) -> int:
        return self._records.delete_records(
            conn,
            ids,
            collection_name,
            cleanup_images=cleanup_images,
            delete_orphan_images=self._delete_orphan_images,
        )

    def delete_by_source_doc_id(
        self,
        collection_name: str,
        source_doc_id: str,
        trace: Optional[Any] = None,
    ) -> int:
        collection = (collection_name or "").strip()
        if not collection:
            raise ValueError("collection_name 不能为空")
        document_id = (source_doc_id or "").strip()
        if not document_id:
            raise ValueError("source_doc_id 不能为空")

        conn = self._get_conn()
        chunk_ids = self._records.find_ids_by_source_doc_id(
            conn,
            collection,
            document_id,
        )
        if not chunk_ids:
            return 0
        return self.delete(ids=chunk_ids, trace=trace, collection_name=collection)

    def _delete_orphan_images(
        self,
        conn: sqlite3.Connection,
        deleted_chunk_ids: List[str],
        collection_name: str,
    ) -> None:
        self._images.delete_orphan_images(
            conn,
            deleted_chunk_ids,
            collection_name,
        )

    def sparse_query(
        self,
        terms: List[str],
        collection_name: str,
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[QueryResult]:
        if not self._write_fts:
            raise RuntimeError("sparse_query 需要 write_fts=True")
        if not terms:
            return []
        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")
        collection = collection_name or self._collection_name
        if not collection:
            raise ValueError("collection_name 不能为空")
        return self._search.sparse_query(
            self._get_conn(),
            terms,
            collection,
            top_k,
        )

    def sparse_index_exists(self, collection_name: str) -> bool:
        if not collection_name or not self._write_fts:
            return False
        return self._search.sparse_index_exists(self._get_conn(), collection_name)

    def list_collections(self) -> List[str]:
        return self._records.list_collections(self._get_conn())

    def get_chunks_by_parent_id(
        self,
        collection_name: str,
        parent_id: str,
        metadata_key: str = "parent_id",
    ) -> List[QueryResult]:
        return self._records.get_chunks_by_parent_id(
            self._get_conn(),
            collection_name,
            parent_id,
            metadata_key,
        )

    def get_image(
        self,
        image_id: str,
        collection_name: str,
    ) -> Optional[tuple]:
        if not self._write_fts:
            return None
        return self._images.get_image(self._get_conn(), image_id, collection_name)

    def get_backend(self) -> str:
        return self._backend

    def get_collection_name(self) -> str:
        return cast(str, self._collection_name)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as exc:
                logger.debug("SQLite 连接关闭时异常: %s", exc)
            self._conn = None
