import json
import sqlite3
from typing import Any, Callable, Dict, List, Optional, cast

from modular_rag_mcp_server.libs.vector_store.base_vector_store import QueryResult, VectorRecord
from modular_rag_mcp_server.libs.vector_store.sqlite_image_repository import SQLiteImageRepository

_METADATA_EXCLUDE = ("image_data",)


def prepare_metadata(metadata: Dict[str, Any]) -> str:
    safe = {
        key: value
        for key, value in metadata.items()
        if key not in _METADATA_EXCLUDE and not isinstance(value, bytes)
    }
    return json.dumps(safe, ensure_ascii=False)


def tokenized_text_for_fts(text: str) -> str:
    from modular_rag_mcp_server.libs.tokenizer import tokenize_for_search

    return " ".join(tokenize_for_search(text))


class SQLiteRecordRepository:
    """读写 chunk 记录及其索引，不管理事务。"""

    def __init__(
        self,
        embedding_dim: int,
        write_fts: bool,
        image_repository: SQLiteImageRepository,
    ) -> None:
        self._dim = int(embedding_dim)
        self._write_fts = bool(write_fts)
        self._images = image_repository

    def validate_records(self, records: List[VectorRecord]) -> None:
        for index, record in enumerate(records):
            if not isinstance(record, VectorRecord):
                raise ValueError(f"记录 {index} 必须是 VectorRecord 类型")
            if not record.id:
                raise ValueError(f"记录 {index} 的 id 不能为空")
            if not record.vector:
                raise ValueError(f"记录 {index} 的 vector 不能为空")
            if not record.text:
                raise ValueError(f"记录 {index} 的 text 不能为空")
            if len(record.vector) != self._dim:
                raise ValueError(
                    f"记录 {index} 向量维度 {len(record.vector)} 与配置 {self._dim} 不一致"
                )

    def write_records(
        self,
        conn: sqlite3.Connection,
        records: List[VectorRecord],
        collection_name: str,
        chunks_for_images: Optional[List[Any]] = None,
        *,
        orphan_candidates: Optional[set[str]] = None,
        upsert_images: Optional[
            Callable[[sqlite3.Connection, List[Any], str], None]
        ] = None,
    ) -> None:
        from sqlite_vec import serialize_float32  # type: ignore[import-untyped]

        candidates = set(orphan_candidates or ())
        for record in records:
            if self._write_fts:
                candidates.update(
                    self._images.collect_image_refs(
                        conn,
                        [record.id],
                        collection_name,
                    )
                )
            self.delete_records(
                conn,
                [record.id],
                collection_name,
                cleanup_images=False,
            )
            conn.execute(
                """
                INSERT INTO chunks (id, collection_name, text, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    record.id,
                    collection_name,
                    record.text,
                    prepare_metadata(record.metadata),
                ),
            )
            rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                """
                INSERT INTO chunks_vec (chunk_rowid, collection_name, embedding)
                VALUES (?, ?, ?)
                """,
                (rowid, collection_name, serialize_float32(record.vector)),
            )
            if self._write_fts:
                conn.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, collection_name, tokenized_text)
                    VALUES (?, ?, ?)
                    """,
                    (
                        record.id,
                        collection_name,
                        tokenized_text_for_fts(record.text),
                    ),
                )
        if self._write_fts:
            if chunks_for_images:
                image_writer = upsert_images or self._images.upsert_images
                image_writer(conn, chunks_for_images, collection_name)
            self._images.delete_unreferenced_images(
                conn,
                candidates,
                collection_name,
            )

    def delete_records(
        self,
        conn: sqlite3.Connection,
        ids: List[str],
        collection_name: str,
        *,
        cleanup_images: bool = True,
        delete_orphan_images: Optional[
            Callable[[sqlite3.Connection, List[str], str], None]
        ] = None,
    ) -> int:
        if not ids:
            return 0
        ids_json = json.dumps(ids, ensure_ascii=False)
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE collection_name = ?
              AND id IN (SELECT value FROM json_each(?))
            """,
            (collection_name, ids_json),
        ).fetchone()
        deleted = int(row[0] if row else 0)
        if deleted == 0:
            return 0

        if self._write_fts:
            if cleanup_images:
                image_cleaner = delete_orphan_images or self._images.delete_orphan_images
                image_cleaner(conn, ids, collection_name)
            conn.execute(
                """
                DELETE FROM chunks_fts
                WHERE collection_name = ?
                  AND chunk_id IN (SELECT value FROM json_each(?))
                """,
                (collection_name, ids_json),
            )
        conn.execute(
            """
            DELETE FROM chunks_vec
            WHERE chunk_rowid IN (
                SELECT rowid
                FROM chunks
                WHERE collection_name = ?
                  AND id IN (SELECT value FROM json_each(?))
            )
            """,
            (collection_name, ids_json),
        )
        conn.execute(
            """
            DELETE FROM chunks
            WHERE collection_name = ?
              AND id IN (SELECT value FROM json_each(?))
            """,
            (collection_name, ids_json),
        )
        return deleted

    def find_ids_by_source_doc_id(
        self,
        conn: sqlite3.Connection,
        collection_name: str,
        source_doc_id: str,
    ) -> List[str]:
        rows = conn.execute(
            """
            SELECT id
            FROM chunks
            WHERE collection_name = ?
              AND json_extract(metadata_json, '$.source_doc_id') = ?
            """,
            (collection_name, source_doc_id),
        ).fetchall()
        return [str(row[0]) for row in rows if row[0]]

    def list_collections(self, conn: sqlite3.Connection) -> List[str]:
        rows = conn.execute(
            "SELECT DISTINCT collection_name FROM chunks ORDER BY collection_name"
        ).fetchall()
        return [row[0] for row in rows]

    def get_chunks_by_parent_id(
        self,
        conn: sqlite3.Connection,
        collection_name: str,
        parent_id: str,
        metadata_key: str = "parent_id",
    ) -> List[QueryResult]:
        key = str(metadata_key or "").strip()
        if not key or '"' in key:
            raise ValueError(f"非法 parent metadata key: {metadata_key!r}")
        rows = conn.execute(
            """
            SELECT id, text, metadata_json
            FROM chunks
            WHERE collection_name = ?
              AND json_extract(metadata_json, ?) = ?
            ORDER BY CAST(json_extract(metadata_json, '$.chunk_index') AS INTEGER)
            """,
            (collection_name, f"$.{key}", parent_id),
        ).fetchall()
        return [
            QueryResult(
                id=row[0],
                score=0.0,
                text=row[1] or "",
                metadata=self._load_metadata(row[2]),
            )
            for row in rows
        ]

    @staticmethod
    def _load_metadata(raw_metadata: Optional[str]) -> Dict[str, Any]:
        if not raw_metadata:
            return {}
        try:
            return cast(Dict[str, Any], json.loads(raw_metadata))
        except json.JSONDecodeError:
            return {}
