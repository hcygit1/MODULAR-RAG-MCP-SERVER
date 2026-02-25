"""Citation generator 单元测试"""
import pytest

from src.core.response.citation_generator import CitationGenerator, generate_citations
from src.libs.vector_store.base_vector_store import QueryResult


def test_generate_citations_empty() -> None:
    """空结果返回空引用"""
    assert generate_citations([]) == []


def test_generate_citations_extracts_fields() -> None:
    """正确提取 source, page, chunk_id, score"""
    results = [
        QueryResult(
            id="chunk_1",
            score=0.95,
            text="some text",
            metadata={"source_path": "/docs/a.pdf", "page": 3, "chunk_index": 0},
        ),
    ]
    citations = generate_citations(results)
    assert len(citations) == 1
    assert citations[0]["source"] == "a.pdf"
    assert citations[0]["page"] == 3
    assert citations[0]["chunk_id"] == "chunk_1"
    assert citations[0]["score"] == 0.95


def test_generate_citations_fallback_source() -> None:
    """无 source_path 时使用 source_doc_id 或 unknown"""
    r = QueryResult(id="c1", score=0.5, text="x", metadata={"source_doc_id": "doc_abc"})
    citations = generate_citations([r])
    assert citations[0]["source"] == "doc_abc"

    r2 = QueryResult(id="c2", score=0.5, text="x", metadata={})
    citations2 = generate_citations([r2])
    assert citations2[0]["source"] == "unknown"


class TestCitationGenerator:
    def test_from_results(self) -> None:
        """类方法 from_results"""
        r = QueryResult(id="c1", score=0.8, text="t", metadata={"source_path": "x.pdf"})
        assert CitationGenerator.from_results([r])[0]["chunk_id"] == "c1"
