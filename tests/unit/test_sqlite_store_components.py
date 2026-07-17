import json
import sqlite3
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import sqlite_vec

from modular_rag_mcp_server.core.settings import VectorStoreConfig
from modular_rag_mcp_server.libs.vector_store.base_vector_store import VectorRecord
from modular_rag_mcp_server.libs.vector_store.sqlite_image_repository import SQLiteImageRepository
from modular_rag_mcp_server.libs.vector_store.sqlite_record_repository import SQLiteRecordRepository
from modular_rag_mcp_server.libs.vector_store.sqlite_schema import SQLiteSchemaManager
from modular_rag_mcp_server.libs.vector_store.sqlite_search import (
    DenseQueryPlan,
    SQLiteSearchService,
)
from modular_rag_mcp_server.libs.vector_store.sqlite_store import SQLiteVectorStore


class TrackingConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commit_calls = 0

    def commit(self):
        self.commit_calls += 1
        return super().commit()


class QueryFailureConnection:
    def __init__(
        self,
        *,
        count_error: sqlite3.OperationalError | None = None,
        query_error: sqlite3.OperationalError | None = None,
    ) -> None:
        self._count_error = count_error
        self._query_error = query_error

    def execute(self, sql, params=()):
        if "SELECT COUNT(*)" in sql:
            if self._count_error is not None:
                raise self._count_error
            return SimpleNamespace(fetchone=lambda: (1,))
        if "FROM chunks_vec" in sql and self._query_error is not None:
            raise self._query_error
        raise AssertionError(f"unexpected SQL: {sql}")


class LateFailingSchemaManager(SQLiteSchemaManager):
    def ensure_tables(self, conn: sqlite3.Connection) -> None:
        super().ensure_tables(conn)
        raise sqlite3.OperationalError("late schema marker")


def _connection(*, row_factory: bool = True) -> TrackingConnection:
    conn = sqlite3.connect(":memory:", factory=TrackingConnection)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn


def _components(*, write_fts: bool = True, row_factory: bool = True):
    conn = _connection(row_factory=row_factory)
    schema = SQLiteSchemaManager(embedding_dim=4, write_fts=write_fts)
    schema.ensure_tables(conn)
    conn.commit()
    images = SQLiteImageRepository()
    records = SQLiteRecordRepository(
        embedding_dim=4,
        write_fts=write_fts,
        image_repository=images,
    )
    search = SQLiteSearchService()
    return conn, images, records, search


def _record(
    record_id: str,
    vector: list[float],
    text: str,
    metadata: dict | None = None,
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        vector=vector,
        text=text,
        metadata=metadata or {},
    )


def _query_store(conn: QueryFailureConnection) -> SQLiteVectorStore:
    store = object.__new__(SQLiteVectorStore)
    store._backend = "sqlite"
    store._collection_name = "search_collection"
    store._dim = 4
    store._search = SQLiteSearchService()
    store._get_conn = lambda: conn
    return store


def _vector_store(tmp_path) -> SQLiteVectorStore:
    config = VectorStoreConfig(
        backend="sqlite",
        persist_path="",
        collection_name="images_collection",
        sqlite_path=str(tmp_path / "components.sqlite"),
        embedding_dim=4,
    )
    return SQLiteVectorStore(config, write_fts=True)


def _image_chunk(image_id: str, image_data: bytes):
    return SimpleNamespace(
        metadata={
            "image_refs": [image_id],
            "image_data": {image_id: image_data},
            "image_metadata": [{"image_id": image_id, "mime_type": "image/png"}],
        }
    )


def test_schema_creates_joint_primary_key_and_optional_tables():
    conn = _connection()
    try:
        manager = SQLiteSchemaManager(embedding_dim=4, write_fts=True)

        manager.ensure_tables(conn)

        primary_key = [
            row["name"]
            for row in sorted(
                conn.execute("PRAGMA table_info(chunks)").fetchall(),
                key=lambda item: int(item["pk"]),
            )
            if int(row["pk"]) > 0
        ]
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        assert primary_key == ["id", "collection_name"]
        assert {"chunks", "chunks_vec", "chunks_fts", "images"} <= tables
    finally:
        conn.close()


def test_schema_migrates_legacy_primary_key_and_preserves_rowid():
    conn = _connection()
    try:
        conn.execute(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                collection_name TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT,
                created_at INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO chunks (
                rowid, id, collection_name, text, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (42, "legacy", "old", "legacy text", "{}", 1),
        )
        conn.commit()
        manager = SQLiteSchemaManager(embedding_dim=4, write_fts=False)

        conn.execute("BEGIN IMMEDIATE")
        manager.ensure_tables(conn)

        primary_key = [
            row["name"]
            for row in sorted(
                conn.execute("PRAGMA table_info(chunks)").fetchall(),
                key=lambda item: int(item["pk"]),
            )
            if int(row["pk"]) > 0
        ]
        row = conn.execute(
            "SELECT rowid, id, collection_name FROM chunks WHERE id = ?",
            ("legacy",),
        ).fetchone()
        assert conn.in_transaction is True
        assert primary_key == ["id", "collection_name"]
        assert tuple(row) == (42, "legacy", "old")
    finally:
        conn.rollback()
        conn.close()


def test_facade_rolls_back_all_new_schema_tables_after_late_ddl_failure():
    conn = _connection()
    store = object.__new__(SQLiteVectorStore)
    store._conn = conn
    store._schema = LateFailingSchemaManager(embedding_dim=4, write_fts=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="late schema marker"):
            store._ensure_tables()

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        assert tables.isdisjoint({"chunks", "chunks_vec", "chunks_fts", "images"})
    finally:
        conn.close()


def test_record_repository_writes_and_deletes_only_target_collection_without_commit():
    conn, _images, repository, _search = _components()
    try:
        initial_commits = conn.commit_calls
        shared_a = _record("shared", [1.0, 0.0, 0.0, 0.0], "alpha")
        shared_b = _record("shared", [0.0, 1.0, 0.0, 0.0], "beta")

        repository.write_records(conn, [shared_a], "collection_a")
        repository.write_records(conn, [shared_b], "collection_b")
        deleted = repository.delete_records(conn, ["shared"], "collection_a")

        rows = conn.execute(
            "SELECT id, collection_name, text FROM chunks ORDER BY collection_name"
        ).fetchall()
        assert deleted == 1
        assert [tuple(row) for row in rows] == [("shared", "collection_b", "beta")]
        assert conn.commit_calls == initial_commits
    finally:
        conn.close()


def test_record_repository_source_lookup_collections_and_parent_query():
    conn, _images, repository, _search = _components(write_fts=False)
    try:
        repository.write_records(
            conn,
            [
                _record(
                    "c2",
                    [1.0, 0.0, 0.0, 0.0],
                    "second",
                    {"source_doc_id": "doc-1", "parent_id": "parent", "chunk_index": 2},
                ),
                _record(
                    "c1",
                    [0.0, 1.0, 0.0, 0.0],
                    "first",
                    {"source_doc_id": "doc-1", "parent_id": "parent", "chunk_index": 1},
                ),
            ],
            "collection_a",
        )
        repository.write_records(
            conn,
            [_record("other", [0.0, 0.0, 1.0, 0.0], "other")],
            "collection_b",
        )

        source_ids = repository.find_ids_by_source_doc_id(conn, "collection_a", "doc-1")
        collections = repository.list_collections(conn)
        children = repository.get_chunks_by_parent_id(conn, "collection_a", "parent")

        assert source_ids == ["c2", "c1"]
        assert collections == ["collection_a", "collection_b"]
        assert [result.id for result in children] == ["c1", "c2"]
        assert children[0].metadata["source_doc_id"] == "doc-1"
        assert all(result.score == 0.0 for result in children)
    finally:
        conn.close()


def test_record_repository_supports_default_tuple_rows():
    conn, _images, repository, _search = _components(
        write_fts=False,
        row_factory=False,
    )
    try:
        repository.write_records(
            conn,
            [
                _record(
                    "tuple-record",
                    [1.0, 0.0, 0.0, 0.0],
                    "tuple",
                    {
                        "source_doc_id": "doc-tuple",
                        "parent_id": "parent-tuple",
                        "chunk_index": 1,
                    },
                )
            ],
            "tuple_collection",
        )

        assert repository.find_ids_by_source_doc_id(
            conn,
            "tuple_collection",
            "doc-tuple",
        ) == ["tuple-record"]
        assert repository.list_collections(conn) == ["tuple_collection"]
        children = repository.get_chunks_by_parent_id(
            conn,
            "tuple_collection",
            "parent-tuple",
        )
        assert [result.id for result in children] == ["tuple-record"]
        assert children[0].metadata["source_doc_id"] == "doc-tuple"
    finally:
        conn.close()


def test_image_repository_writes_reads_and_cleans_orphans_without_commit():
    conn, images, repository, _search = _components()
    try:
        initial_commits = conn.commit_calls
        referenced = SimpleNamespace(
            metadata={
                "image_refs": ["shared", "orphan"],
                "image_data": {"shared": b"shared", "orphan": b"orphan"},
                "image_metadata": [
                    {"image_id": "shared", "mime_type": "image/jpeg", "page": 1},
                    {"image_id": "orphan", "mime_type": "image/png"},
                ],
            }
        )
        repository.write_records(
            conn,
            [
                _record(
                    "remove",
                    [1.0, 0.0, 0.0, 0.0],
                    "remove",
                    {"image_refs": ["shared", "orphan"]},
                ),
                _record(
                    "keep",
                    [0.0, 1.0, 0.0, 0.0],
                    "keep",
                    {"image_refs": ["shared"]},
                ),
            ],
            "images_collection",
            chunks_for_images=[referenced],
        )

        loaded = images.get_image(conn, "shared", "images_collection")
        images.delete_orphan_images(conn, ["remove"], "images_collection")

        remaining = conn.execute(
            "SELECT id FROM images WHERE collection_name = ? ORDER BY id",
            ("images_collection",),
        ).fetchall()
        assert loaded == (b"shared", "image/jpeg")
        assert [row[0] for row in remaining] == ["shared"]
        assert conn.commit_calls == initial_commits
    finally:
        conn.close()


def test_upsert_preserves_existing_image_when_new_record_keeps_reference(tmp_path):
    store = _vector_store(tmp_path)
    try:
        store.upsert(
            [
                _record(
                    "same",
                    [1.0, 0.0, 0.0, 0.0],
                    "old",
                    {"image_refs": ["img-1"]},
                )
            ],
            collection_name="images_collection",
            chunks_for_images=[_image_chunk("img-1", b"original")],
        )

        store.upsert(
            [
                _record(
                    "same",
                    [0.0, 1.0, 0.0, 0.0],
                    "new",
                    {"image_refs": ["img-1"]},
                )
            ],
            collection_name="images_collection",
        )

        assert store.get_image("img-1", "images_collection") == (
            b"original",
            "image/png",
        )
    finally:
        store.close()


def test_replace_preserves_existing_image_when_new_document_keeps_reference(tmp_path):
    store = _vector_store(tmp_path)
    try:
        store.upsert(
            [
                _record(
                    "old-chunk",
                    [1.0, 0.0, 0.0, 0.0],
                    "old",
                    {"source_doc_id": "doc-1", "image_refs": ["img-1"]},
                )
            ],
            collection_name="images_collection",
            chunks_for_images=[_image_chunk("img-1", b"original")],
        )

        store.replace_document_records(
            source_doc_id="doc-1",
            records=[
                _record(
                    "new-chunk",
                    [0.0, 1.0, 0.0, 0.0],
                    "new",
                    {"source_doc_id": "doc-1", "image_refs": ["img-1"]},
                )
            ],
            collection_name="images_collection",
        )

        assert store.get_image("img-1", "images_collection") == (
            b"original",
            "image/png",
        )
    finally:
        store.close()


def test_upsert_deletes_existing_image_when_new_record_removes_reference(tmp_path):
    store = _vector_store(tmp_path)
    try:
        store.upsert(
            [
                _record(
                    "same",
                    [1.0, 0.0, 0.0, 0.0],
                    "old",
                    {"image_refs": ["img-1"]},
                )
            ],
            collection_name="images_collection",
            chunks_for_images=[_image_chunk("img-1", b"original")],
        )

        store.upsert(
            [_record("same", [0.0, 1.0, 0.0, 0.0], "new")],
            collection_name="images_collection",
        )

        assert store.get_image("img-1", "images_collection") is None
    finally:
        store.close()


def test_search_builds_parameterized_equality_and_in_filters():
    sql, params = SQLiteSearchService.build_filter_sql(
        {"source": "guide", "kind": {"$in": ["api", "tutorial"]}}
    )

    assert sql == (
        "WHERE json_extract(c.metadata_json, ?) = ? AND "
        "json_extract(c.metadata_json, ?) IN (SELECT value FROM json_each(?))"
    )
    assert params == ["$.source", "guide", "$.kind", json.dumps(["api", "tutorial"])]


def test_facade_preserves_candidate_count_operational_error():
    marker = sqlite3.OperationalError("candidate count marker")
    store = _query_store(QueryFailureConnection(count_error=marker))

    with pytest.raises(sqlite3.OperationalError) as caught:
        store.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            top_k=1,
            filters={"source": "guide"},
            collection_name="search_collection",
        )

    assert caught.value is marker
    assert caught.value.__cause__ is None


def test_facade_wraps_final_vector_query_operational_error():
    marker = sqlite3.OperationalError("vector query marker")
    store = _query_store(QueryFailureConnection(query_error=marker))

    with pytest.raises(RuntimeError, match="SQLite query 失败") as caught:
        store.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            top_k=1,
            collection_name="search_collection",
        )

    assert caught.value.__cause__ is marker


def test_dense_query_plan_is_immutable():
    service = SQLiteSearchService()
    plan = service.prepare_dense_query(
        QueryFailureConnection(),
        top_k=1,
        collection_name="search_collection",
        filters={"source": "guide"},
    )

    assert isinstance(plan, DenseQueryPlan)
    assert plan.candidate_k == 1
    with pytest.raises(FrozenInstanceError):
        plan.candidate_k = 2


def test_facade_dense_query_uses_compatibility_filter_helper():
    marker = sqlite3.OperationalError("vector query marker")
    store = _query_store(QueryFailureConnection(query_error=marker))
    calls = []
    store._build_filter_sql = lambda filters: calls.append(filters) or ("", [])

    with pytest.raises(RuntimeError, match="SQLite query 失败"):
        store.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            top_k=1,
            filters={"source": "guide"},
            collection_name="search_collection",
        )

    assert calls == [{"source": "guide"}]


def test_facade_record_helpers_use_compatibility_image_helpers(tmp_path):
    store = _vector_store(tmp_path)
    try:
        conn = store._get_conn()
        upsert_calls = []
        delete_calls = []
        store._upsert_images = lambda connection, chunks, collection: upsert_calls.append(
            (connection, chunks, collection)
        )
        chunk = _image_chunk("img-helper", b"helper")

        store._write_records(
            conn,
            [
                _record(
                    "helper-record",
                    [1.0, 0.0, 0.0, 0.0],
                    "helper",
                    {"image_refs": ["img-helper"]},
                )
            ],
            "images_collection",
            [chunk],
        )
        store._delete_orphan_images = (
            lambda connection, ids, collection: delete_calls.append(
                (connection, ids, collection)
            )
        )
        store._delete_records(conn, ["helper-record"], "images_collection")

        assert upsert_calls == [(conn, [chunk], "images_collection")]
        assert delete_calls == [(conn, ["helper-record"], "images_collection")]
    finally:
        store.close()


def test_search_maps_dense_and_sparse_results():
    conn, _images, repository, search = _components()
    try:
        repository.write_records(
            conn,
            [
                _record(
                    "python",
                    [1.0, 0.0, 0.0, 0.0],
                    "Python retrieval",
                    {"source": "guide"},
                ),
                _record(
                    "other",
                    [0.0, 1.0, 0.0, 0.0],
                    "Other content",
                    {"source": "notes"},
                ),
            ],
            "search_collection",
        )

        dense = search.query(
            conn,
            vector=[1.0, 0.0, 0.0, 0.0],
            top_k=1,
            collection_name="search_collection",
            filters={"source": "guide"},
        )
        sparse = search.sparse_query(
            conn,
            terms=["Python"],
            collection_name="search_collection",
            top_k=2,
        )

        assert [(result.id, result.text) for result in dense] == [
            ("python", "Python retrieval")
        ]
        assert dense[0].metadata == {"source": "guide"}
        assert [result.id for result in sparse] == ["python"]
        assert sparse[0].metadata == {"source": "guide"}
    finally:
        conn.close()


def test_search_maps_dense_and_sparse_default_tuple_rows():
    conn, _images, repository, search = _components(row_factory=False)
    try:
        repository.write_records(
            conn,
            [
                _record(
                    "tuple-search",
                    [1.0, 0.0, 0.0, 0.0],
                    "Tuple search",
                    {"source": "tuple"},
                )
            ],
            "tuple_collection",
        )

        dense = search.query(
            conn,
            vector=[1.0, 0.0, 0.0, 0.0],
            top_k=1,
            collection_name="tuple_collection",
        )
        sparse = search.sparse_query(
            conn,
            terms=["Tuple"],
            collection_name="tuple_collection",
            top_k=1,
        )

        assert [result.id for result in dense] == ["tuple-search"]
        assert dense[0].metadata == {"source": "tuple"}
        assert [result.id for result in sparse] == ["tuple-search"]
        assert sparse[0].metadata == {"source": "tuple"}
    finally:
        conn.close()


@pytest.mark.parametrize(
    "filters",
    [
        {'bad"key': "value"},
        {"kind": {"$not_supported": ["value"]}},
    ],
)
def test_search_rejects_unsupported_filters(filters):
    with pytest.raises(ValueError):
        SQLiteSearchService.build_filter_sql(filters)
