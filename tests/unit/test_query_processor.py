"""
QueryProcessor 单元测试

验证关键词提取与 filters 解析功能。
"""
import pytest

from src.core.query_engine.query_processor import (
    ProcessedQuery,
    QueryProcessor,
)


class TestQueryProcessorBasic:
    """基础功能测试"""

    def test_process_returns_keywords_and_filters(self) -> None:
        """对输入 query 输出 keywords（非空）和 filters（dict）"""
        processor = QueryProcessor()
        result = processor.process("RAG system architecture")

        assert isinstance(result, ProcessedQuery)
        assert isinstance(result.keywords, list)
        assert isinstance(result.filters, dict)
        assert result.original_query == "RAG system architecture"
        # 典型查询应有关键词
        assert len(result.keywords) > 0
        assert "rag" in result.keywords
        assert "system" in result.keywords
        assert "architecture" in result.keywords

    def test_stopwords_filtered(self) -> None:
        """停用词被过滤"""
        processor = QueryProcessor()
        result = processor.process("the and a system design")

        assert "the" not in result.keywords
        assert "and" not in result.keywords
        assert "a" not in result.keywords
        assert "system" in result.keywords
        assert "design" in result.keywords

    def test_filters_is_dict(self) -> None:
        """filters 为 dict 类型"""
        processor = QueryProcessor()
        result = processor.process("test query")
        assert result.filters == {}

    def test_keywords_deduplicated(self) -> None:
        """关键词去重"""
        processor = QueryProcessor()
        result = processor.process("rag rag system system")
        assert result.keywords.count("rag") == 1
        assert result.keywords.count("system") == 1


class TestQueryProcessorChinese:
    """中文支持测试"""

    def test_chinese_keywords_extracted(self) -> None:
        """中文关键词被正确提取"""
        processor = QueryProcessor()
        result = processor.process("RAG 系统架构设计")

        assert len(result.keywords) > 0
        assert "rag" in result.keywords or "RAG".lower() in result.keywords
        # 中文非停用词应保留
        assert any(
            "\u4e00" <= c <= "\u9fff" for kw in result.keywords for c in kw
        ) or "系统" in result.keywords or "架构" in result.keywords

    def test_chinese_stopwords_filtered(self) -> None:
        """中文停用词被过滤"""
        processor = QueryProcessor()
        result = processor.process("这是一个系统")
        # "是"、"的" 等为停用词
        assert "是" not in result.keywords
        assert "这" not in result.keywords


class TestQueryProcessorEdgeCases:
    """边界情况测试"""

    def test_empty_query(self) -> None:
        """空查询返回空 keywords"""
        processor = QueryProcessor()
        result = processor.process("")
        assert result.keywords == []
        assert result.filters == {}
        assert result.original_query == ""

    def test_whitespace_only_query(self) -> None:
        """仅空白字符的查询"""
        processor = QueryProcessor()
        result = processor.process("   ")
        assert result.keywords == []
        assert result.filters == {}

    def test_all_stopwords_query(self) -> None:
        """全部为停用词的查询"""
        processor = QueryProcessor()
        result = processor.process("the and a is of")
        assert result.keywords == []
        assert isinstance(result.filters, dict)

    def test_none_query_treated_as_empty(self) -> None:
        """None 作为空字符串处理"""
        processor = QueryProcessor()
        result = processor.process(None)  # type: ignore
        assert result.keywords == []
        assert result.original_query == ""


class TestQueryProcessorMinTermLength:
    """min_term_length 配置测试"""

    def test_min_term_length_filters_short_tokens(self) -> None:
        """min_term_length 过滤过短的 token"""
        processor = QueryProcessor(min_term_length=2)
        result = processor.process("a bc d ef")
        assert "a" not in result.keywords
        assert "d" not in result.keywords
        assert "bc" in result.keywords
        assert "ef" in result.keywords


class TestQueryProcessorCustomStopwords:
    """自定义停用词测试"""

    def test_custom_stopwords(self) -> None:
        """自定义停用词生效"""
        processor = QueryProcessor(stopwords={"custom", "stop"})
        result = processor.process("custom stop word")
        assert "custom" not in result.keywords
        assert "stop" not in result.keywords
        assert "word" in result.keywords


class TestProcessedQuery:
    """ProcessedQuery 数据类测试"""

    def test_keywords_must_be_list(self) -> None:
        """keywords 必须为 list"""
        with pytest.raises(TypeError, match="keywords 必须是 list"):
            ProcessedQuery(
                keywords="not-a-list",  # type: ignore
                filters={},
                original_query="",
            )

    def test_filters_must_be_dict(self) -> None:
        """filters 必须为 dict"""
        with pytest.raises(TypeError, match="filters 必须是 dict"):
            ProcessedQuery(
                keywords=[],
                filters="not-a-dict",  # type: ignore
                original_query="",
            )
