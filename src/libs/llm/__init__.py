"""
LLM 模块

提供 LLM 抽象接口和工厂实现。
"""
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.fake_llm import FakeLLM

__all__ = ["BaseLLM", "LLMFactory", "FakeLLM"]
