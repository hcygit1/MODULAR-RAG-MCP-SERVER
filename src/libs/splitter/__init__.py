"""
Splitter 模块

提供文本切分抽象接口和工厂实现。
"""
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.splitter.fake_splitter import FakeSplitter

__all__ = ["BaseSplitter", "SplitterFactory", "FakeSplitter"]
