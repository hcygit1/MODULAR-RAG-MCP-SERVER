"""
共享分词模块

为 SparseEncoder、SparseRetriever、QueryProcessor 提供一致的中英文分词逻辑。
"""
from src.libs.tokenizer.jieba_tokenizer import tokenize

__all__ = ["tokenize"]
