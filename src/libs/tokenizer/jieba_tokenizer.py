"""
jieba 分词实现

统一的中英文分词，供 SparseEncoder、SparseRetriever、QueryProcessor 共用。
"""
from typing import List, Optional, Set


def _get_default_stopwords() -> Set[str]:
    """默认停用词集合（与 SparseEncoder / QueryProcessor 保持一致）"""
    english_stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "will", "with", "this", "but", "they", "have",
        "had", "what", "said", "each", "which", "their", "time", "if",
        "up", "out", "many", "then", "them", "these", "so", "some", "her",
        "would", "make", "like", "into", "him", "two", "more",
        "very", "after", "words", "long", "than", "first", "been", "call",
        "who", "oil", "sit", "now", "find", "down", "day", "did", "get",
        "come", "made", "may", "part",
    }
    chinese_stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这",
    }
    additional_stopwords = {
        "over", "under", "through", "during", "before", "after",
        "above", "below", "between", "among", "within", "without",
    }
    return english_stopwords | chinese_stopwords | additional_stopwords


def tokenize(
    text: str,
    stopwords: Optional[Set[str]] = None,
    min_length: int = 1,
) -> List[str]:
    """
    使用 jieba 对文本分词，支持中英文。

    Args:
        text: 输入文本
        stopwords: 停用词集合，None 时使用默认
        min_length: 最小 token 长度，低于此长度的 token 被过滤

    Returns:
        过滤后的 token 列表（保持顺序，去重由调用方决定）
    """
    import jieba

    if not text or not text.strip():
        return []

    _stopwords = stopwords if stopwords is not None else _get_default_stopwords()
    text_lower = text.strip().lower()

    # jieba.lcut 对中文按词切分，英文/数字保持完整
    tokens = jieba.lcut(text_lower)

    filtered = [
        t for t in tokens
        if len(t) >= min_length
        and t not in _stopwords
        and t.strip()
    ]
    return filtered
