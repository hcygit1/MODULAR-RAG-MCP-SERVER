"""
FTS5 BM25 Indexer 实现

使用 SQLite FTS5 虚拟表实现 BM25 全文检索，与 SQLiteVectorStore 共用同一数据库。
分词采用 jieba.cut_for_search，与索引构建保持一致。
"""
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ingestion.models import Chunk
from src.libs.tokenizer import tokenize_for_search

logger = logging.getLogger(__name__)

_FTS5_TABLE = "chunks_fts"


def _tokenized_text(chunk: Chunk) -> str:
    """对 chunk 文本进行分词，用于 FTS5 索引。"""
    return " ".join(tokenize_for_search(chunk.text))


def _escape_fts5_term(term: str) -> str:
    """转义 FTS5 查询特殊字符。"""
    for char in ('"', "'", " ", "-", "(", ")", "*"):
        term = term.replace(char, " ")
    return term.strip()


class FTS5BM25Indexer:
    """
    FTS5 BM25 Indexer

    使用 chunks_fts 虚拟表存储 tokenized_text，与 chunks/chunks_vec 同库。
    text 和 metadata 从 chunks 表 JOIN 获取，本索引器不存 metadata。
    """

    def __init__(self, sqlite_path: str) -> None:
        if not sqlite_path:
            raise ValueError("sqlite_path 不能为空")
        self._path = Path(sqlite_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._collection_name: Optional[str] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_table(self) -> None:
        conn = self._get_conn()
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS5_TABLE} USING fts5(
                chunk_id UNINDEXED,
                collection_name UNINDEXED,
                tokenized_text
            )
        """)
        conn.commit()

    def build(
        self,
        chunks: List[Chunk],
        sparse_vectors: List[Dict[str, float]],
        collection_name: str,
        trace: Optional[Any] = None,
    ) -> None:
        """构建 FTS5 索引，清空该 collection 的旧数据后写入新 chunks。"""
        if not chunks:
            raise ValueError("chunks 列表不能为空")
        if len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"chunks 数量 ({len(chunks)}) 与 sparse_vectors 数量 ({len(sparse_vectors)}) 不一致"
            )
        if not collection_name:
            raise ValueError("collection_name 不能为空")

        self._collection_name = collection_name
        self._ensure_table()
        conn = self._get_conn()
        conn.execute(
            f"DELETE FROM {_FTS5_TABLE} WHERE collection_name = ?",
            (collection_name,),
        )
        for chunk in chunks:
            tt = _tokenized_text(chunk)
            conn.execute(
                f"INSERT INTO {_FTS5_TABLE} (chunk_id, collection_name, tokenized_text) VALUES (?, ?, ?)",
                (chunk.id, collection_name, tt),
            )
        conn.commit()

    def merge(
        self,
        chunks: List[Chunk],
        sparse_vectors: List[Dict[str, float]],
        collection_name: str,
        trace: Optional[Any] = None,
    ) -> None:
        """将新 chunks 增量合并。已存在的 chunk_id 先删除再插入。"""
        if not chunks:
            raise ValueError("chunks 列表不能为空")
        if len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"chunks 数量 ({len(chunks)}) 与 sparse_vectors 数量 ({len(sparse_vectors)}) 不一致"
            )
        if not collection_name:
            raise ValueError("collection_name 不能为空")

        self._collection_name = collection_name
        self._ensure_table()
        conn = self._get_conn()
        chunk_ids = [c.id for c in chunks]
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(
            f"DELETE FROM {_FTS5_TABLE} WHERE collection_name = ? AND chunk_id IN ({placeholders})",
            (collection_name, *chunk_ids),
        )
        for chunk in chunks:
            tt = _tokenized_text(chunk)
            conn.execute(
                f"INSERT INTO {_FTS5_TABLE} (chunk_id, collection_name, tokenized_text) VALUES (?, ?, ?)",
                (chunk.id, collection_name, tt),
            )
        conn.commit()

    def save(self) -> None:
        """FTS5 数据已持久化到 SQLite，无需额外 save。"""
        pass

    def load(self, collection_name: str) -> None:
        """设置当前 collection，用于后续 query。"""
        if not collection_name:
            raise ValueError("collection_name 不能为空")
        self._collection_name = collection_name

    def query(
        self,
        terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Tuple[str, float]]:
        """
        查询 FTS5 索引，返回 (chunk_id, bm25_score) 列表。

        FTS5 的 bm25 值越小越好，此处取负值转为越大越好以兼容 BM25Indexer 接口。
        """
        if not terms:
            return []
        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")
        if not self._collection_name:
            raise RuntimeError("索引未加载，无法查询。请先调用 load()")

        conn = self._get_conn()
        self._ensure_table()

        safe_terms = [_escape_fts5_term(t) for t in terms if t and t.strip()]
        if not safe_terms:
            return []
        query_str = " OR ".join(f'"{t}"' for t in safe_terms)

        try:
            rows = conn.execute(
                f"""
                SELECT chunk_id, bm25({_FTS5_TABLE}) AS rk
                FROM {_FTS5_TABLE}
                WHERE {_FTS5_TABLE} MATCH ? AND collection_name = ?
                ORDER BY rk
                LIMIT ?
                """,
                (query_str, self._collection_name, top_k),
            ).fetchall()
        except sqlite3.OperationalError as e:
            if "syntax error" in str(e).lower() or "malformed" in str(e).lower():
                return []
            raise

        # FTS5 bm25 越小越相关，取负值使分数越大越好
        return [(row["chunk_id"], -float(row["rk"] or 0)) for row in rows]

    def index_exists(self, collection_name: str) -> bool:
        """检查该 collection 在 chunks_fts 中是否有数据。"""
        if not collection_name:
            return False
        self._ensure_table()
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT 1 FROM {_FTS5_TABLE} WHERE collection_name = ? LIMIT 1",
            (collection_name,),
        ).fetchone()
        return row is not None

    def get_collection_name(self) -> Optional[str]:
        return self._collection_name

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as e:
                logger.debug("FTS5 连接关闭异常: %s", e)
            self._conn = None
