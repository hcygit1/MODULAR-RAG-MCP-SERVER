"""
jieba 分词模块单元测试
"""
import pytest

from src.libs.tokenizer import tokenize
from src.libs.tokenizer.jieba_tokenizer import _get_default_stopwords


class TestJiebaTokenizerBasic:
    """基础功能测试"""

    def test_empty_text_returns_empty(self) -> None:
        """空文本返回空列表"""
        assert tokenize("") == []
        assert tokenize("   ") == []
        assert tokenize("\n\t") == []

    def test_english_tokenization(self) -> None:
        """英文分词（停用词 what/is 被过滤，rag 保留）"""
        result = tokenize("what is RAG")
        assert "what" not in result  # 停用词
        assert "is" not in result  # 停用词
        assert "rag" in result

    def test_chinese_tokenization(self) -> None:
        """中文分词（jieba 可能得到 毕业设计、指导老师 等词）"""
        result = tokenize("毕业设计指导老师")
        assert len(result) >= 1
        all_tokens = "".join(result)
        assert "毕业" in all_tokens or "设计" in all_tokens or "指导" in all_tokens or "老师" in all_tokens

    def test_mixed_chinese_english(self) -> None:
        """中英混合"""
        result = tokenize("氢能车辆租赁APP的设计与实现")
        assert "氢能" in result or "车辆" in result or "租赁" in result
        assert "app" in result or "设计" in result


class TestJiebaTokenizerStopwords:
    """停用词过滤"""

    def test_english_stopwords_filtered(self) -> None:
        """英文停用词过滤"""
        result = tokenize("the quick brown fox")
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_chinese_stopwords_filtered(self) -> None:
        """中文停用词过滤"""
        result = tokenize("这是一个系统")
        assert "是" not in result
        assert "这" not in result


class TestJiebaTokenizerMinLength:
    """min_length 参数"""

    def test_min_length_filters_short(self) -> None:
        """min_length 过滤过短 token"""
        result = tokenize("a bc d ef", min_length=2)
        assert "a" not in result
        assert "d" not in result
        assert "bc" in result
        assert "ef" in result


class TestJiebaTokenizerCustomStopwords:
    """自定义停用词"""

    def test_custom_stopwords(self) -> None:
        """自定义停用词生效"""
        result = tokenize("custom stop word", stopwords={"custom", "stop"})
        assert "custom" not in result
        assert "stop" not in result
        assert "word" in result


class TestDefaultStopwords:
    """默认停用词"""

    def test_default_stopwords_contains_common(self) -> None:
        """默认停用词包含常见词"""
        sw = _get_default_stopwords()
        assert "the" in sw
        assert "的" in sw
        assert "是" in sw
