"""
冒烟测试：验证关键包可以正常导入

此测试确保项目的基本结构正确，所有关键模块可以正常导入。
这是最基础的验证，用于快速发现项目配置问题。
"""
import pytest


def test_import_src():
    """验证 src 包可以正常导入"""
    import src
    assert src is not None
    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"


def test_import_core():
    """验证 core 模块可以正常导入"""
    import src.core
    assert src.core is not None


def test_import_ingestion():
    """验证 ingestion 模块可以正常导入"""
    import src.ingestion
    assert src.ingestion is not None


def test_import_libs():
    """验证 libs 模块可以正常导入"""
    import src.libs
    assert src.libs is not None


def test_import_mcp_server():
    """验证 mcp_server 模块可以正常导入"""
    import src.mcp_server
    assert src.mcp_server is not None


def test_import_observability():
    """验证 observability 模块可以正常导入"""
    import src.observability
    assert src.observability is not None


def test_import_core_submodules():
    """验证 core 子模块可以正常导入"""
    import src.core.query_engine
    import src.core.response
    import src.core.trace
    assert src.core.query_engine is not None
    assert src.core.response is not None
    assert src.core.trace is not None


def test_import_ingestion_submodules():
    """验证 ingestion 子模块可以正常导入"""
    import src.ingestion.embedding
    import src.ingestion.storage
    import src.ingestion.transform
    assert src.ingestion.embedding is not None
    assert src.ingestion.storage is not None
    assert src.ingestion.transform is not None


def test_import_libs_submodules():
    """验证 libs 子模块可以正常导入"""
    import src.libs.embedding
    import src.libs.evaluator
    import src.libs.llm
    import src.libs.loader
    import src.libs.reranker
    import src.libs.splitter
    import src.libs.vector_store
    assert src.libs.embedding is not None
    assert src.libs.evaluator is not None
    assert src.libs.llm is not None
    assert src.libs.loader is not None
    assert src.libs.reranker is not None
    assert src.libs.splitter is not None
    assert src.libs.vector_store is not None


def test_import_mcp_server_submodules():
    """验证 mcp_server 子模块可以正常导入"""
    import src.mcp_server.tools
    assert src.mcp_server.tools is not None


def test_import_observability_submodules():
    """验证 observability 子模块可以正常导入"""
    import src.observability.dashboard
    import src.observability.evaluation
    assert src.observability.dashboard is not None
    assert src.observability.evaluation is not None
