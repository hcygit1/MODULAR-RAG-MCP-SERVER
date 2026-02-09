"""
Sparse Encoder 实现

对 Chunks 建立 BM25 所需统计，生成稀疏向量（Term Weights）：
- 分词和统计：计算每个 chunk 的 term frequency
- Term Weights：生成词到权重的映射结构
- 输出格式：便于后续 bm25_indexer 使用
"""
import re
from typing import List, Dict, Optional, Any
from collections import Counter

from src.ingestion.models import Chunk


class SparseEncoder:
    """
    Sparse Encoder 实现
    
    为 Chunk 列表生成 BM25 所需的 term weights 结构。
    输出格式可用于后续的 bm25_indexer。
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        min_term_length: int = 1,
        stopwords: Optional[set] = None
    ):
        """
        初始化 SparseEncoder
        
        Args:
            k1: BM25 参数 k1（控制 term frequency 饱和度），默认 1.5
            b: BM25 参数 b（控制文档长度归一化），默认 0.75
            min_term_length: 最小词长度，默认 1
            stopwords: 停用词集合（可选），如果为 None 则使用默认停用词
        """
        self._k1 = k1
        self._b = b
        self._min_term_length = min_term_length
        self._stopwords = stopwords or self._get_default_stopwords()
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None
    ) -> List[Dict[str, float]]:
        """
        批量将 Chunks 转换为稀疏向量（Term Weights）
        
        Args:
            chunks: Chunk 对象列表
            trace: 追踪上下文（可选）
        
        Returns:
            List[Dict[str, float]]: Term weights 列表，每个 Chunk 对应一个字典
                                   - 字典键为 term（词），值为权重（float）
                                   - 长度与 chunks 相同
        
        Raises:
            ValueError: 当 chunks 为空时
        """
        if not chunks:
            raise ValueError("chunks 列表不能为空")
        
        # 计算文档集合的统计信息（用于 IDF 计算）
        doc_freqs = self._compute_document_frequencies(chunks)
        avg_doc_length = self._compute_average_document_length(chunks)
        
        # 为每个 chunk 生成 term weights
        term_weights_list = []
        for chunk in chunks:
            term_weights = self._compute_term_weights(
                chunk.text,
                doc_freqs,
                avg_doc_length,
                len(chunks)
            )
            term_weights_list.append(term_weights)
        
        return term_weights_list
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词：将文本转换为词列表
        
        Args:
            text: 输入文本
        
        Returns:
            List[str]: 词列表
        """
        if not text:
            return []
        
        # 转换为小写
        text_lower = text.lower()
        
        # 使用正则表达式提取单词（支持中英文）
        # 匹配：字母、数字、中文字符
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b|[\u4e00-\u9fff]+', text_lower)
        
        # 过滤：长度、停用词
        filtered_tokens = [
            token for token in tokens
            if len(token) >= self._min_term_length
            and token not in self._stopwords
        ]
        
        return filtered_tokens
    
    def _compute_term_frequency(self, text: str) -> Dict[str, int]:
        """
        计算词频（Term Frequency）
        
        Args:
            text: 输入文本
        
        Returns:
            Dict[str, int]: 词到频率的映射
        """
        tokens = self._tokenize(text)
        return dict(Counter(tokens))
    
    def _compute_document_frequencies(
        self,
        chunks: List[Chunk]
    ) -> Dict[str, int]:
        """
        计算文档频率（Document Frequency）：每个词出现在多少个文档中
        
        Args:
            chunks: Chunk 列表
        
        Returns:
            Dict[str, int]: 词到文档频率的映射
        """
        doc_freqs = {}
        
        for chunk in chunks:
            # 获取该 chunk 的唯一词集合
            unique_terms = set(self._tokenize(chunk.text))
            
            # 统计每个词出现在多少个文档中
            for term in unique_terms:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        
        return doc_freqs
    
    def _compute_average_document_length(
        self,
        chunks: List[Chunk]
    ) -> float:
        """
        计算平均文档长度
        
        Args:
            chunks: Chunk 列表
        
        Returns:
            float: 平均文档长度（词数）
        """
        if not chunks:
            return 0.0
        
        total_length = sum(len(self._tokenize(chunk.text)) for chunk in chunks)
        return total_length / len(chunks)
    
    def _compute_term_weights(
        self,
        text: str,
        doc_freqs: Dict[str, int],
        avg_doc_length: float,
        total_docs: int
    ) -> Dict[str, float]:
        """
        计算 BM25 term weights
        
        Args:
            text: Chunk 文本
            doc_freqs: 文档频率字典
            avg_doc_length: 平均文档长度
            total_docs: 文档总数
        
        Returns:
            Dict[str, float]: Term weights 字典
        """
        if not text or not text.strip():
            # 空文本返回空字典（明确行为）
            return {}
        
        # 计算词频
        term_freqs = self._compute_term_frequency(text)
        doc_length = len(self._tokenize(text))
        
        # 计算归一化文档长度
        normalized_length = doc_length / avg_doc_length if avg_doc_length > 0 else 1.0
        
        # 计算 BM25 权重
        term_weights = {}
        for term, tf in term_freqs.items():
            # 计算 IDF (Inverse Document Frequency)
            df = doc_freqs.get(term, 0)
            if df == 0:
                # 词不在任何文档中（不应该发生，但防御性处理）
                idf = 0.0
            else:
                # 标准 IDF 公式：log((N - df + 0.5) / (df + 0.5))
                idf = self._compute_idf(total_docs, df)
            
            # 计算 BM25 权重
            # BM25 公式：idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
            k = self._k1 * (1 - self._b + self._b * normalized_length)
            weight = idf * (tf * (self._k1 + 1)) / (tf + k)
            
            term_weights[term] = weight
        
        return term_weights
    
    def _compute_idf(self, total_docs: int, doc_freq: int) -> float:
        """
        计算 IDF (Inverse Document Frequency)
        
        Args:
            total_docs: 文档总数
            doc_freq: 文档频率（包含该词的文档数）
        
        Returns:
            float: IDF 值（可能为负数，当词出现在大多数文档中时）
        """
        if doc_freq == 0 or total_docs == 0:
            return 0.0
        
        # 标准 IDF 公式：log((N - df + 0.5) / (df + 0.5))
        # 使用自然对数
        # 注意：当 df > N/2 时，IDF 可能为负数，这是正常的
        import math
        idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        return idf
    
    def _get_default_stopwords(self) -> set:
        """
        获取默认停用词集合
        
        Returns:
            set: 停用词集合
        """
        # 英文常用停用词
        english_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
            'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
            'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
            'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
        
        # 中文常用停用词
        chinese_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这'
        }
        
        # 添加更多英文停用词
        additional_stopwords = {
            'over', 'under', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'within', 'without'
        }
        
        return english_stopwords | chinese_stopwords | additional_stopwords
