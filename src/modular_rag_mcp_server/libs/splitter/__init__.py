"""
Splitter 模块

提供文本切分抽象接口和工厂实现。
"""
from modular_rag_mcp_server.libs.splitter.base_splitter import BaseSplitter
from modular_rag_mcp_server.libs.splitter.fake_splitter import FakeSplitter

# B7.5 阶段实现
from modular_rag_mcp_server.libs.splitter.recursive_splitter import RecursiveSplitter
from modular_rag_mcp_server.libs.splitter.splitter_factory import SplitterFactory

__all__ = [
    "BaseSplitter",
    "SplitterFactory",
    "FakeSplitter",
    "RecursiveSplitter"
]
