import re
import sqlite3
from typing import Any, Optional

_EMBEDDING_DIMENSION_PATTERN = re.compile(
    r"\bembedding\s+float\s*\[\s*(\d+)\s*\]",
    re.IGNORECASE,
)


class SQLiteSchemaManager:
    """创建和迁移 SQLiteVectorStore 使用的表，不管理事务。"""

    def __init__(self, embedding_dim: int, write_fts: bool = False) -> None:
        self._dim = int(embedding_dim)
        self._write_fts = bool(write_fts)

    @staticmethod
    def create_chunks_table(
        conn: sqlite3.Connection,
        table_name: str = "chunks",
    ) -> None:
        conn.execute(
            f"""
            CREATE TABLE {table_name} (
                id TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT,
                created_at INTEGER DEFAULT (strftime('%s','now')),
                PRIMARY KEY (id, collection_name)
            )
            """
        )

    @staticmethod
    def _column_value(row: Any, name: str, index: int) -> Any:
        try:
            return row[name]
        except (IndexError, TypeError):
            return row[index]

    def chunks_table_needs_migration(self, conn: sqlite3.Connection) -> bool:
        columns = conn.execute("PRAGMA table_info(chunks)").fetchall()
        if not columns:
            return False
        primary_key = [
            self._column_value(row, "name", 1)
            for row in sorted(
                columns,
                key=lambda item: int(self._column_value(item, "pk", 5)),
            )
            if int(self._column_value(row, "pk", 5)) > 0
        ]
        return primary_key != ["id", "collection_name"]

    def ensure_chunks_table(self, conn: sqlite3.Connection) -> None:
        columns = conn.execute("PRAGMA table_info(chunks)").fetchall()
        if not columns:
            self.create_chunks_table(conn)
            return
        if not self.chunks_table_needs_migration(conn):
            return

        self.create_chunks_table(conn, "chunks_migrated")
        conn.execute(
            """
            INSERT INTO chunks_migrated (
                rowid,
                id,
                collection_name,
                text,
                metadata_json,
                created_at
            )
            SELECT rowid, id, collection_name, text, metadata_json, created_at
            FROM chunks
            """
        )
        conn.execute("DROP TABLE chunks")
        conn.execute("ALTER TABLE chunks_migrated RENAME TO chunks")

    @staticmethod
    def embedding_dimension(conn: sqlite3.Connection) -> Optional[int]:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'chunks_vec'"
        ).fetchone()
        if row is None:
            return None
        ddl = str(row[0] or "")
        match = _EMBEDDING_DIMENSION_PATTERN.search(ddl)
        if match is None:
            raise RuntimeError("无法识别现有 chunks_vec 的 embedding 维度")
        return int(match.group(1))

    def ensure_vector_table(self, conn: sqlite3.Connection) -> None:
        existing_dim = self.embedding_dimension(conn)
        if existing_dim == self._dim:
            return
        if existing_dim is not None:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            chunk_count = int(row[0] if row else 0)
            if chunk_count:
                raise RuntimeError(
                    "SQLite embedding_dim 不匹配: "
                    f"数据库={existing_dim}, 配置={self._dim}。"
                    f"现有 {chunk_count} 条 chunk，无法自动迁移；"
                    "请备份数据库后清空并重新入库。"
                )
            conn.execute("DROP TABLE chunks_vec")

        vec0_ddl = f"""
            CREATE VIRTUAL TABLE chunks_vec USING vec0(
                chunk_rowid INTEGER PRIMARY KEY,
                embedding float[{self._dim}] distance_metric=cosine,
                collection_name TEXT
            )
            """.strip()
        conn.execute(vec0_ddl)

    def ensure_tables(self, conn: sqlite3.Connection) -> None:
        self.ensure_chunks_table(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection_name)"
        )

        if self._dim <= 0 or self._dim > 32768:
            raise ValueError(f"非法 embedding_dim: {self._dim}")
        self.ensure_vector_table(conn)

        if not self._write_fts:
            return
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                collection_name UNINDEXED,
                tokenized_text
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                image_data BLOB NOT NULL,
                mime_type TEXT DEFAULT 'image/png',
                metadata_json TEXT,
                created_at INTEGER DEFAULT (strftime('%s','now')),
                PRIMARY KEY (id, collection_name)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_collection ON images(collection_name)"
        )
