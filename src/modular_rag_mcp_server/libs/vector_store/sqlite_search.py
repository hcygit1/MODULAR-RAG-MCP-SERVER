import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, cast

from modular_rag_mcp_server.libs.vector_store.base_vector_store import QueryResult


@dataclass(frozen=True)
class DenseQueryPlan:
    collection_name: str
    top_k: int
    filter_sql: str
    filter_params: tuple[Any, ...]
    candidate_k: int


class SQLiteSearchService:
    """执行 SQLite dense/sparse 检索并映射结果，不管理事务。"""

    @staticmethod
    def build_filter_sql(
        filters: Optional[Dict[str, Any]],
    ) -> tuple[str, List[Any]]:
        if not filters:
            return "", []

        clauses: List[str] = []
        params: List[Any] = []
        for raw_key, value in filters.items():
            key = str(raw_key or "").strip()
            if not key or '"' in key:
                raise ValueError(f"非法 metadata filter key: {raw_key!r}")
            json_path = f"$.{key}"
            if isinstance(value, dict):
                if set(value) != {"$in"} or not isinstance(value["$in"], list):
                    raise ValueError(
                        f"SQLite metadata filter 仅支持 equality 和 $in: {key}"
                    )
                clauses.append(
                    "json_extract(c.metadata_json, ?) "
                    "IN (SELECT value FROM json_each(?))"
                )
                params.extend(
                    [json_path, json.dumps(value["$in"], ensure_ascii=False)]
                )
            else:
                clauses.append("json_extract(c.metadata_json, ?) = ?")
                params.extend([json_path, value])
        return "WHERE " + " AND ".join(clauses), params

    def prepare_dense_query(
        self,
        conn: sqlite3.Connection,
        top_k: int,
        collection_name: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        filter_builder: Optional[
            Callable[
                [Optional[Dict[str, Any]]],
                tuple[str, List[Any]],
            ]
        ] = None,
    ) -> DenseQueryPlan:
        builder = filter_builder or self.build_filter_sql
        filter_sql, filter_params = builder(filters)
        return DenseQueryPlan(
            collection_name=collection_name,
            top_k=top_k,
            filter_sql=filter_sql,
            filter_params=tuple(filter_params),
            candidate_k=top_k,
        )

    def execute_dense_query(
        self,
        conn: sqlite3.Connection,
        vector: List[float],
        plan: DenseQueryPlan,
    ) -> List[QueryResult]:
        from sqlite_vec import serialize_float32  # type: ignore[import-untyped]

        vector_blob = serialize_float32(vector)
        if plan.filter_sql:
            if not plan.filter_sql.startswith("WHERE "):
                raise ValueError("SQLite metadata filter SQL 必须以 WHERE 开头")
            filter_predicate = plan.filter_sql.removeprefix("WHERE ")
            rows = conn.execute(
                f"""
                SELECT
                    c.id,
                    c.text,
                    c.metadata_json,
                    vec_distance_cosine(v.embedding, ?) AS distance
                FROM chunks c
                JOIN chunks_vec v ON v.chunk_rowid = c.rowid
                WHERE c.collection_name = ? AND ({filter_predicate})
                ORDER BY distance
                LIMIT ?
                """,
                (
                    vector_blob,
                    plan.collection_name,
                    *plan.filter_params,
                    plan.top_k,
                ),
            ).fetchall()
            return self._map_dense_rows(rows)

        rows = conn.execute(
            f"""
            SELECT c.id, c.text, c.metadata_json, v.distance
            FROM (
                SELECT chunk_rowid, distance
                FROM chunks_vec
                WHERE embedding MATCH ? AND k = ? AND collection_name = ?
            ) v
            JOIN chunks c ON c.rowid = v.chunk_rowid
            {plan.filter_sql}
            ORDER BY v.distance
            LIMIT ?
            """,
            (
                vector_blob,
                plan.candidate_k,
                plan.collection_name,
                *plan.filter_params,
                plan.top_k,
            ),
        ).fetchall()
        return self._map_dense_rows(rows)

    def _map_dense_rows(self, rows: List[Any]) -> List[QueryResult]:
        return [
            QueryResult(
                id=row[0],
                score=(
                    1.0 - float(row[3])
                    if row[3] is not None
                    else 0.0
                ),
                text=row[1] or "",
                metadata=self._load_metadata(row[2]),
            )
            for row in rows
        ]

    def query(
        self,
        conn: sqlite3.Connection,
        vector: List[float],
        top_k: int,
        collection_name: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        plan = self.prepare_dense_query(
            conn,
            top_k,
            collection_name,
            filters,
        )
        return self.execute_dense_query(conn, vector, plan)

    def sparse_query(
        self,
        conn: sqlite3.Connection,
        terms: List[str],
        collection_name: str,
        top_k: int,
    ) -> List[QueryResult]:
        from modular_rag_mcp_server.ingestion.storage.fts5_bm25_indexer import _escape_fts5_term

        safe_terms = [_escape_fts5_term(term) for term in terms if term and str(term).strip()]
        if not safe_terms:
            return []
        query_string = " OR ".join(f'"{term}"' for term in safe_terms)
        try:
            rows = conn.execute(
                """
                SELECT c.id, c.text, c.metadata_json, fts.rk
                FROM (
                    SELECT chunk_id, bm25(chunks_fts) AS rk
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ? AND collection_name = ?
                    ORDER BY rk
                    LIMIT ?
                ) fts
                JOIN chunks c ON c.id = fts.chunk_id AND c.collection_name = ?
                """,
                (query_string, collection_name, top_k, collection_name),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "syntax error" in str(exc).lower() or "malformed" in str(exc).lower():
                return []
            raise
        return [
            QueryResult(
                id=row[0],
                score=-float(row[3] or 0),
                text=row[1] or "",
                metadata=self._load_metadata(row[2]),
            )
            for row in rows
        ]

    @staticmethod
    def sparse_index_exists(
        conn: sqlite3.Connection,
        collection_name: str,
    ) -> bool:
        row = conn.execute(
            "SELECT 1 FROM chunks_fts WHERE collection_name = ? LIMIT 1",
            (collection_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _load_metadata(raw_metadata: Optional[str]) -> Dict[str, Any]:
        if not raw_metadata:
            return {}
        try:
            return cast(Dict[str, Any], json.loads(raw_metadata))
        except json.JSONDecodeError:
            return {}
