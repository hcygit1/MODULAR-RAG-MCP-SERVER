"""
SQLite VectorStore 单元测试
"""
import tempfile
from pathlib import Path

import pytest

from src.core.settings import VectorStoreConfig
from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
)
from src.libs.vector_store.sqlite_store import SQLiteVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


def _make_config(tmp_path: Path, dim: int = 4) -> VectorStoreConfig:
    return VectorStoreConfig(
        backend="sqlite",
        persist_path="./data/db/qdrant",
        collection_name="test_coll",
        sqlite_path=str(tmp_path / "rag.sqlite"),
        embedding_dim=dim,
    )


@pytest.fixture
def sqlite_store(tmp_path):
    config = _make_config(tmp_path, dim=4)
    store = SQLiteVectorStore(config)
    yield store
    store.close()


def test_sqlite_store_implements_base(sqlite_store):
    assert isinstance(sqlite_store, BaseVectorStore)
    assert sqlite_store.get_backend() == "sqlite"
    assert sqlite_store.get_collection_name() == "test_coll"


def test_sqlite_store_upsert_and_query(sqlite_store):
    records = [
        VectorRecord(id="c1", vector=[1.0, 0.0, 0.0, 0.0], text="doc 1", metadata={"x": 1}),
        VectorRecord(id="c2", vector=[0.9, 0.1, 0.0, 0.0], text="doc 2", metadata={"x": 2}),
        VectorRecord(id="c3", vector=[0.0, 0.0, 1.0, 0.0], text="doc 3", metadata={"x": 3}),
    ]
    sqlite_store.upsert(records, collection_name="test_coll")

    results = sqlite_store.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        top_k=2,
        collection_name="test_coll",
    )
    assert len(results) == 2
    assert results[0].id == "c1"
    assert results[0].text == "doc 1"
    assert results[0].metadata.get("x") == 1
    assert results[1].id == "c2"


def test_sqlite_store_delete(sqlite_store):
    records = [
        VectorRecord(id="c1", vector=[1.0, 0.0, 0.0, 0.0], text="doc 1", metadata={}),
    ]
    sqlite_store.upsert(records, collection_name="test_coll")

    results = sqlite_store.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        collection_name="test_coll",
    )
    assert len(results) == 1

    deleted = sqlite_store.delete(["c1"], collection_name="test_coll")
    assert deleted == 1

    results = sqlite_store.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        collection_name="test_coll",
    )
    assert len(results) == 0


def test_sqlite_store_upsert_idempotent(sqlite_store):
    records = [
        VectorRecord(id="c1", vector=[1.0, 0.0, 0.0, 0.0], text="v1", metadata={}),
    ]
    sqlite_store.upsert(records, collection_name="test_coll")
    sqlite_store.upsert(records, collection_name="test_coll")
    records[0] = VectorRecord(id="c1", vector=[0.9, 0.1, 0.0, 0.0], text="v2", metadata={})
    sqlite_store.upsert(records, collection_name="test_coll")

    results = sqlite_store.query(
        vector=[0.9, 0.1, 0.0, 0.0],
        top_k=1,
        collection_name="test_coll",
    )
    assert len(results) == 1
    assert results[0].text == "v2"


def test_sqlite_store_write_fts_upsert_and_delete(tmp_path):
    """Phase 3: write_fts=True 时单事务写入 chunks/chunks_vec/chunks_fts，delete 同步删除"""
    config = _make_config(tmp_path, dim=4)
    store = SQLiteVectorStore(config, write_fts=True)
    try:
        records = [
            VectorRecord(id="c1", vector=[1.0, 0.0, 0.0, 0.0], text="Python programming", metadata={}),
            VectorRecord(id="c2", vector=[0.0, 1.0, 0.0, 0.0], text="Machine learning", metadata={}),
        ]
        store.upsert(records, collection_name="test_coll")

        conn = store._get_conn()
        rows = conn.execute(
            "SELECT chunk_id, tokenized_text FROM chunks_fts WHERE collection_name = ?",
            ("test_coll",),
        ).fetchall()
        assert len(rows) == 2
        chunk_ids = {r[0] for r in rows}
        assert "c1" in chunk_ids and "c2" in chunk_ids

        store.delete(["c1"], collection_name="test_coll")
        rows = conn.execute(
            "SELECT chunk_id FROM chunks_fts WHERE collection_name = ?",
            ("test_coll",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "c2"

        results = store.query(vector=[0.0, 1.0, 0.0, 0.0], top_k=5, collection_name="test_coll")
        assert len(results) == 1
        assert results[0].id == "c2"
    finally:
        store.close()


def test_sqlite_store_upsert_with_images(tmp_path):
    """Phase 4: chunks_for_images 时同期写入 images 表"""
    from src.ingestion.models import Chunk

    config = _make_config(tmp_path, dim=4)
    store = SQLiteVectorStore(config, write_fts=True)
    try:
        chunks = [
            Chunk(
                id="c1",
                text="doc with image",
                metadata={
                    "image_refs": ["img_1"],
                    "image_data": {"img_1": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"},
                    "image_metadata": [{"image_id": "img_1", "mime_type": "image/png"}],
                },
            ),
        ]
        records = [
            VectorRecord(id="c1", vector=[1.0, 0.0, 0.0, 0.0], text="doc with image", metadata={}),
        ]
        store.upsert(records, collection_name="test_coll", chunks_for_images=chunks)

        conn = store._get_conn()
        row = conn.execute(
            "SELECT id, image_data, mime_type FROM images WHERE collection_name = ?",
            ("test_coll",),
        ).fetchone()
        assert row is not None
        assert row[0] == "img_1"
        assert row[1] == b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        assert row[2] == "image/png"
    finally:
        store.close()


def test_sqlite_store_delete_orphan_images(tmp_path):
    """Phase 4: 删除 chunk 时移除不再被引用的图片"""
    from src.ingestion.models import Chunk

    config = _make_config(tmp_path, dim=4)
    store = SQLiteVectorStore(config, write_fts=True)
    try:
        chunk = Chunk(
            id="c1",
            text="doc1",
            metadata={"image_refs": ["img_solo"], "image_data": {"img_solo": b"x"}, "image_metadata": []},
        )
        records = [
            VectorRecord(
                id="c1",
                vector=[1.0, 0.0, 0.0, 0.0],
                text="doc1",
                metadata={"image_refs": ["img_solo"]},
            )
        ]
        store.upsert(records, collection_name="test_coll", chunks_for_images=[chunk])

        conn = store._get_conn()
        assert conn.execute("SELECT COUNT(*) FROM images WHERE collection_name = ?", ("test_coll",)).fetchone()[0] == 1

        store.delete(["c1"], collection_name="test_coll")
        assert conn.execute("SELECT COUNT(*) FROM images WHERE collection_name = ?", ("test_coll",)).fetchone()[0] == 0
    finally:
        store.close()


def test_factory_creates_sqlite_store(tmp_path):
    """当 backend=sqlite 时，工厂应返回 SQLiteVectorStore"""
    from src.core.settings import Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig
    from src.core.settings import RetrievalConfig, RerankConfig, EvaluationConfig
    from src.core.settings import ObservabilityConfig, IngestionConfig, MinerUConfig
    from src.core.settings import LoggingConfig, DashboardConfig

    config = VectorStoreConfig(
        backend="sqlite",
        persist_path="./data/db/qdrant",
        collection_name="test",
        sqlite_path=str(tmp_path / "rag.sqlite"),
        embedding_dim=4,
    )
    # 构造最小 Settings（仅 vector_store 用于工厂）
    settings = Settings(
        llm=LLMConfig(provider="qwen", model="test"),
        vision_llm=VisionLLMConfig(provider="qwen", model="test"),
        embedding=EmbeddingConfig(provider="qwen", model="test"),
        vector_store=config,
        retrieval=RetrievalConfig(
            sparse_backend="bm25", fusion_algorithm="rrf",
            top_k_dense=20, top_k_sparse=20, top_k_final=10,
        ),
        rerank=RerankConfig(backend="none", model="", top_m=30, timeout_seconds=5),
        evaluation=EvaluationConfig(backends=[], golden_test_set=""),
        observability=ObservabilityConfig(
            enabled=True,
            logging=LoggingConfig(log_file="", log_level="INFO"),
            detail_level="standard",
            dashboard=DashboardConfig(enabled=False, port=8501),
        ),
        ingestion=IngestionConfig(
            chunk_size=512, chunk_overlap=50,
            enable_llm_refinement=False, enable_metadata_enrichment=False,
            enable_image_captioning=False, batch_size=32,
        ),
        mineru=MinerUConfig(api_token="", model_version="", poll_interval_seconds=5, poll_timeout_seconds=600),
    )
    store = VectorStoreFactory.create(settings)
    assert isinstance(store, SQLiteVectorStore)
    store.close()
