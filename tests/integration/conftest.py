"""
Integration 测试共享 fixtures

供 test_retrieval_pipeline、test_mcp_server 等复用。
"""
import shutil
import tempfile

import pytest

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import RerankerOrchestrator
from src.core.query_engine.retrieval_pipeline import RetrievalPipeline
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.models import Chunk
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.fake_embedding import FakeEmbedding
from src.libs.reranker.none_reranker import NoneReranker
from src.libs.vector_store.base_vector_store import VectorRecord
from src.libs.vector_store.fake_vector_store import FakeVectorStore


@pytest.fixture
def temp_bm25_dir():
    """创建临时 BM25 索引目录"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_chunks():
    """测试用 chunks"""
    return [
        Chunk(
            id="chunk_1",
            text="Python is a programming language for data science",
            metadata={"source_path": "/path/to/doc1.pdf", "chunk_index": 0, "page": 1},
        ),
        Chunk(
            id="chunk_2",
            text="Machine learning is a subset of artificial intelligence",
            metadata={"source_path": "/path/to/doc1.pdf", "chunk_index": 1, "page": 2},
        ),
        Chunk(
            id="chunk_3",
            text="Python and RAG are used for retrieval augmented generation",
            metadata={"source_path": "/path/to/doc2.pdf", "chunk_index": 0, "page": 1},
        ),
    ]


@pytest.fixture
def indexed_fixtures(temp_bm25_dir, sample_chunks):
    """构建 Dense + Sparse 双路 fixtures"""
    encoder = SparseEncoder()
    sparse_vectors = encoder.encode(sample_chunks)
    indexer = BM25Indexer(base_path=temp_bm25_dir)
    indexer.build(sample_chunks, sparse_vectors, collection_name="test_collection")
    indexer.save()

    embedding = FakeEmbedding(dimension=16)
    vector_store = FakeVectorStore(collection_name="test_collection")

    texts = [c.text for c in sample_chunks]
    vectors = embedding.embed(texts)

    records = [
        VectorRecord(id=c.id, vector=vectors[i], text=c.text, metadata=c.metadata)
        for i, c in enumerate(sample_chunks)
    ]
    vector_store.upsert(records)

    return {
        "vector_store": vector_store,
        "embedding": embedding,
        "bm25_path": temp_bm25_dir,
        "collection_name": "test_collection",
    }


@pytest.fixture
def retrieval_pipeline(indexed_fixtures):
    """构建完整 RetrievalPipeline"""
    fixtures = indexed_fixtures
    dense = DenseRetriever(
        embedding=fixtures["embedding"],
        vector_store=fixtures["vector_store"],
    )
    sparse = SparseRetriever(
        base_path=fixtures["bm25_path"],
        collection_name=fixtures["collection_name"],
    )
    hybrid = HybridSearch(dense_retriever=dense, sparse_retriever=sparse)
    reranker = RerankerOrchestrator(backend=NoneReranker())

    return RetrievalPipeline(
        query_processor=QueryProcessor(),
        hybrid_search=hybrid,
        reranker=reranker,
    )
