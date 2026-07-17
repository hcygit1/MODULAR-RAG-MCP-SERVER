"""
冒烟测试：验证关键包可以正常导入

此测试确保项目的基本结构正确，所有关键模块可以正常导入。
这是最基础的验证，用于快速发现项目配置问题。
"""


def test_import_formal_package():
    """验证正式顶层包可以正常导入"""
    import modular_rag_mcp_server

    assert modular_rag_mcp_server is not None
    assert hasattr(modular_rag_mcp_server, "__version__")
    assert modular_rag_mcp_server.__version__ == "0.1.0"


def test_import_core():
    """验证 core 模块可以正常导入"""
    import modular_rag_mcp_server.core
    assert modular_rag_mcp_server.core is not None


def test_import_ingestion():
    """验证 ingestion 模块可以正常导入"""
    import modular_rag_mcp_server.ingestion
    assert modular_rag_mcp_server.ingestion is not None


def test_import_libs():
    """验证 libs 模块可以正常导入"""
    import modular_rag_mcp_server.libs
    assert modular_rag_mcp_server.libs is not None


def test_import_mcp_server():
    """验证 mcp_server 模块可以正常导入"""
    import modular_rag_mcp_server.mcp_server
    assert modular_rag_mcp_server.mcp_server is not None


def test_import_observability():
    """验证 observability 模块可以正常导入"""
    import modular_rag_mcp_server.observability
    assert modular_rag_mcp_server.observability is not None


def test_import_core_submodules():
    """验证 core 子模块可以正常导入"""
    import modular_rag_mcp_server.core.query_engine
    import modular_rag_mcp_server.core.response
    import modular_rag_mcp_server.core.trace
    assert modular_rag_mcp_server.core.query_engine is not None
    assert modular_rag_mcp_server.core.response is not None
    assert modular_rag_mcp_server.core.trace is not None


def test_import_ingestion_submodules():
    """验证 ingestion 子模块可以正常导入"""
    import modular_rag_mcp_server.ingestion.embedding
    import modular_rag_mcp_server.ingestion.storage
    import modular_rag_mcp_server.ingestion.transform
    assert modular_rag_mcp_server.ingestion.embedding is not None
    assert modular_rag_mcp_server.ingestion.storage is not None
    assert modular_rag_mcp_server.ingestion.transform is not None


def test_import_libs_submodules():
    """验证 libs 子模块可以正常导入"""
    import modular_rag_mcp_server.libs.embedding
    import modular_rag_mcp_server.libs.evaluator
    import modular_rag_mcp_server.libs.llm
    import modular_rag_mcp_server.libs.loader
    import modular_rag_mcp_server.libs.reranker
    import modular_rag_mcp_server.libs.splitter
    import modular_rag_mcp_server.libs.vector_store
    assert modular_rag_mcp_server.libs.embedding is not None
    assert modular_rag_mcp_server.libs.evaluator is not None
    assert modular_rag_mcp_server.libs.llm is not None
    assert modular_rag_mcp_server.libs.loader is not None
    assert modular_rag_mcp_server.libs.reranker is not None
    assert modular_rag_mcp_server.libs.splitter is not None
    assert modular_rag_mcp_server.libs.vector_store is not None


def test_import_mcp_server_submodules():
    """验证 mcp_server 子模块可以正常导入"""
    import modular_rag_mcp_server.mcp_server.tools
    assert modular_rag_mcp_server.mcp_server.tools is not None


def test_import_observability_submodules():
    """验证 observability 子模块可以正常导入"""
    import modular_rag_mcp_server.observability.dashboard
    import modular_rag_mcp_server.observability.evaluation
    assert modular_rag_mcp_server.observability.dashboard is not None
    assert modular_rag_mcp_server.observability.evaluation is not None
