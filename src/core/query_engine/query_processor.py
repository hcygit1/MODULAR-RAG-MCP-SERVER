"""
Query Processor 实现

负责查询预处理：
- 关键词提取：从 query 中提取关键实体与动词（去停用词），用于稀疏检索（BM25）
- Filters 解析：解析通用 filters 结构（如 collection、doc_type 等）
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from src.libs.tokenizer import tokenize as tokenize_text


@dataclass
class ProcessedQuery:
    """查询预处理结果"""

    keywords: List[str]
    """提取的关键词列表，用于稀疏检索（BM25）"""

    filters: Dict[str, Any]
    """通用过滤条件，如 collection、doc_type、language 等"""

    original_query: str
    """原始查询字符串"""

    def __post_init__(self) -> None:
        if not isinstance(self.keywords, list):
            raise TypeError("keywords 必须是 list")
        if not isinstance(self.filters, dict):
            raise TypeError("filters 必须是 dict")


class QueryProcessor:
    """
    Query Processor

    对用户查询进行预处理，提取关键词并解析过滤条件。
    输出格式与 SparseEncoder 的分词策略保持一致，便于 BM25 检索匹配。
    """

    def __init__(
        self,
        min_term_length: int = 1,
        stopwords: Optional[Set[str]] = None,
    ) -> None:
        """
        初始化 QueryProcessor

        Args:
            min_term_length: 最小词长度，短于此长度的 token 被过滤
            stopwords: 停用词集合，为 None 时使用默认停用词
        """
        self._min_term_length = min_term_length
        self._stopwords = stopwords or self._get_default_stopwords()

    def process(self, query: str) -> ProcessedQuery:
        """
        处理查询，提取关键词与 filters

        Args:
            query: 用户原始查询字符串

        Returns:
            ProcessedQuery: 包含 keywords（非空）、filters（dict）
        """
        if query is None:
            query = ""
        query = query.strip()

        keywords = self._extract_keywords(query)
        filters = self._parse_filters(query)

        return ProcessedQuery(
            keywords=keywords,
            filters=filters,
            original_query=query,
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """
        从查询中提取关键词（使用与 SparseEncoder 一致的 jieba 分词）
        """
        keywords = tokenize_text(
            query,
            stopwords=self._stopwords,
            min_length=self._min_term_length,
        )
        # 去重保持顺序
        seen: Set[str] = set()
        result: List[str] = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                result.append(k)
        return result

    def _parse_filters(self, query: str) -> Dict[str, any]:
        """
        解析 filters 结构

        当前为占位实现，返回空 dict。
        后续 D7 RetrievalPipeline 集成 MetadataFilter 时，可实现 collection、doc_type 等解析。

        Args:
            query: 原始查询（可用于解析结构化约束）

        Returns:
            Dict[str, Any]: 过滤条件字典
        """
        # D1 占位：空实现
        return {}

    def _get_default_stopwords(self) -> Set[str]:
        """获取默认停用词集合（与共享 tokenizer 一致）"""
        from src.libs.tokenizer.jieba_tokenizer import _get_default_stopwords
        return _get_default_stopwords()
