"""
MCP Server 集成测试

验证 Server 能完成 initialize；stderr 有日志，stdout 仅含 MCP 消息（不污染）。
以子进程方式启动 server，发送 initialize 请求，断言响应与日志分离。
E2：验证 query_knowledge_hub tool 返回 Markdown + citations。
"""
import json
import subprocess
import sys

import pytest


def _start_server_subprocess():
    """启动 MCP Server 子进程，返回 Popen 实例。"""
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=None,
    )
    return proc


def _send_initialize(proc: subprocess.Popen) -> tuple[str, str]:
    """
    向 server 发送 initialize 请求，关闭 stdin，收集 stdout 和 stderr。

    Returns:
        (stdout_str, stderr_str)
    """
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.1.0"},
        },
    }
    req_line = json.dumps(request) + "\n"
    stdout_data, stderr_data = proc.communicate(input=req_line, timeout=5)
    return stdout_data, stderr_data


class TestMCPServerE1:
    """E1 验收：MCP Server 骨架与 Stdio 约束"""

    def test_initialize_returns_valid_response(self) -> None:
        """发送 initialize 能返回正确的 serverInfo 和 capabilities"""
        proc = _start_server_subprocess()
        stdout_data, stderr_data = _send_initialize(proc)

        assert proc.returncode == 0, f"Server exited with {proc.returncode}, stderr: {stderr_data}"

        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        assert len(lines) >= 1, f"Expected at least 1 response line, got: {stdout_data}"

        resp = json.loads(lines[0])
        assert resp.get("jsonrpc") == "2.0"
        assert resp.get("id") == 1
        assert "result" in resp

        result = resp["result"]
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "modular-rag-mcp-server"
        assert "version" in result["serverInfo"]  # FastMCP 返回 SDK 版本
        assert "capabilities" in result
        assert "tools" in result["capabilities"]

    def test_stderr_has_logs_stdout_not_polluted(self) -> None:
        """stderr 有日志输出，stdout 不包含日志格式（仅 MCP 消息）"""
        proc = _start_server_subprocess()
        stdout_data, stderr_data = _send_initialize(proc)

        assert proc.returncode == 0

        # stdout 每行应为合法 JSON-RPC 消息（不含日志污染）
        for line in stdout_data.split("\n"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert "jsonrpc" in obj, f"stdout 应为 JSON-RPC: {line[:80]}"

        # stderr 有输出（FastMCP 可能输出较少，仅需非空）
        assert len(stderr_data) > 0, "stderr 应有输出"


def _mcp_handshake() -> str:
    """MCP 握手：initialize + notifications/initialized。FastMCP 要求先初始化才能调用 tools/list 等。"""
    init_req = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.1.0"},
        },
    }
    notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    return json.dumps(init_req) + "\n" + json.dumps(notif) + "\n"


def _send_request(proc: subprocess.Popen, request: dict) -> tuple[str, str]:
    """发送单条 JSON-RPC 请求，返回 (stdout, stderr)。"""
    req_line = json.dumps(request) + "\n"
    stdout_data, stderr_data = proc.communicate(input=req_line, timeout=5)
    return stdout_data, stderr_data


def _send_with_handshake(proc: subprocess.Popen, request: dict) -> tuple[str, str]:
    """发送 MCP 握手 + 请求，返回 (stdout, stderr)。响应最后一行为 request 的 result。"""
    payload = _mcp_handshake() + json.dumps(request) + "\n"
    stdout_data, stderr_data = proc.communicate(input=payload, timeout=5)
    return stdout_data, stderr_data


class TestMCPServerE15:
    """E1.5 验收：tools/list、tools/call 协议层"""

    def test_tools_list_returns_schema(self) -> None:
        """发送 tools/list 能返回 tools 数组（可为空）"""
        proc = _start_server_subprocess()
        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        stdout_data, _ = _send_with_handshake(proc, request)
        assert proc.returncode == 0
        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        assert len(lines) >= 2, "应至少有 initialize 与 tools/list 两条响应"
        resp = json.loads(lines[-1])
        assert resp.get("jsonrpc") == "2.0"
        assert resp.get("id") == 2
        assert "result" in resp
        assert "tools" in resp["result"]
        assert isinstance(resp["result"]["tools"], list)

    def test_tools_call_unknown_tool_returns_error_content(self) -> None:
        """tools/call 未知工具时返回 isError=True 的 result"""
        proc = _start_server_subprocess()
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
        stdout_data, _ = _send_with_handshake(proc, request)
        assert proc.returncode == 0
        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        resp = json.loads(lines[-1])
        assert "result" in resp
        result = resp["result"]
        assert result.get("isError") is True
        assert "content" in result
        assert len(result["content"]) >= 1
        err_text = result["content"][0].get("text", "")
        assert "unknown" in err_text.lower() or "not found" in err_text.lower() or "nonexistent" in err_text.lower()

    def test_tools_call_missing_name_returns_invalid_params(self) -> None:
        """tools/call 缺 name 时返回 JSON-RPC -32602"""
        proc = _start_server_subprocess()
        request = {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {}}
        stdout_data, _ = _send_with_handshake(proc, request)
        assert proc.returncode == 0
        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        resp = json.loads(lines[-1])
        assert "error" in resp
        assert resp["error"]["code"] == -32602


class TestMCPServerE2:
    """E2 验收：query_knowledge_hub Tool"""

    def test_tools_list_includes_query_knowledge_hub(self) -> None:
        """tools/list 包含 query_knowledge_hub schema"""
        proc = _start_server_subprocess()
        request = {"jsonrpc": "2.0", "id": 10, "method": "tools/list"}
        stdout_data, _ = _send_with_handshake(proc, request)
        assert proc.returncode == 0
        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        resp = json.loads(lines[-1])
        tools = resp["result"]["tools"]
        names = [t["name"] for t in tools]
        assert "query_knowledge_hub" in names
        qkh = next(t for t in tools if t["name"] == "query_knowledge_hub")
        assert "query" in (qkh.get("inputSchema", {}).get("required") or [])
        assert "query" in (qkh.get("inputSchema", {}).get("properties") or {})

    def test_tools_list_includes_list_collections(self) -> None:
        """tools/list 包含 list_collections schema"""
        proc = _start_server_subprocess()
        request = {"jsonrpc": "2.0", "id": 11, "method": "tools/list"}
        stdout_data, _ = _send_with_handshake(proc, request)
        assert proc.returncode == 0
        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        resp = json.loads(lines[-1])
        names = [t["name"] for t in resp["result"]["tools"]]
        assert "list_collections" in names

    def test_tools_list_includes_get_document_summary(self) -> None:
        """tools/list 包含 get_document_summary schema"""
        proc = _start_server_subprocess()
        request = {"jsonrpc": "2.0", "id": 12, "method": "tools/list"}
        stdout_data, _ = _send_with_handshake(proc, request)
        assert proc.returncode == 0
        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        resp = json.loads(lines[-1])
        names = [t["name"] for t in resp["result"]["tools"]]
        assert "get_document_summary" in names

    def test_query_knowledge_hub_returns_markdown_and_citations(
        self, retrieval_pipeline, indexed_fixtures
    ) -> None:
        """query_knowledge_hub 返回 content[0] 为 Markdown，structuredContent.citations 含 source/page/chunk_id/score"""
        from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub, set_pipeline

        set_pipeline(retrieval_pipeline)

        result = query_knowledge_hub(
            query="python data science",
            collection_name=indexed_fixtures["collection_name"],
            top_k=5,
        )

        assert result.isError is False
        assert len(result.content) >= 1
        assert result.content[0].type == "text"
        assert len(result.content[0].text) > 0
        # Markdown 应包含片段标题或来源
        text = result.content[0].text
        assert "片段" in text or "来源" in text or "Python" in text

        assert result.structuredContent is not None
        assert "citations" in result.structuredContent
        citations = result.structuredContent["citations"]
        assert len(citations) >= 1
        for c in citations:
            assert "source" in c
            assert "chunk_id" in c
            assert "score" in c

    def test_query_knowledge_hub_returns_image_content_when_chunk_has_image_refs(
        self, retrieval_pipeline, indexed_fixtures, tmp_path
    ) -> None:
        """当 chunk 含 image_refs 时，content 中应包含 type=image 的 ImageContent"""
        import base64
        import json
        from pathlib import Path

        from src.ingestion.models import Chunk
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.ingestion.embedding.sparse_encoder import SparseEncoder
        from src.libs.embedding.fake_embedding import FakeEmbedding
        from src.libs.reranker.none_reranker import NoneReranker
        from src.libs.vector_store.base_vector_store import VectorRecord
        from src.libs.vector_store.fake_vector_store import FakeVectorStore
        from src.core.query_engine.dense_retriever import DenseRetriever
        from src.core.query_engine.hybrid_search import HybridSearch
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.reranker import RerankerOrchestrator
        from src.core.query_engine.retrieval_pipeline import RetrievalPipeline
        from src.core.query_engine.sparse_retriever import SparseRetriever
        from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub, set_pipeline, set_config_path

        collection = indexed_fixtures["collection_name"]
        images_base = tmp_path / "images"
        coll_dir = images_base / collection
        coll_dir.mkdir(parents=True)
        img_id = "test_doc_page_0_img_0"
        img_path = coll_dir / f"{img_id}.png"
        img_bytes = b"\x89PNG\r\n\x1a\n"
        img_path.write_bytes(img_bytes)
        index_data = {
            "collection_name": collection,
            "images": {
                img_id: {
                    "image_id": img_id,
                    "file_path": str(img_path),
                    "mime_type": "image/png",
                },
            },
        }
        (coll_dir / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

        chunk_with_image = Chunk(
            id="chunk_with_img",
            text="Chart showing data science growth [IMAGE: test_doc_page_0_img_0]",
            metadata={
                "source_path": "doc.pdf",
                "chunk_index": 0,
                "page": 1,
                "image_refs": [img_id],
            },
        )
        encoder = SparseEncoder()
        sparse = encoder.encode([chunk_with_image])
        indexer = BM25Indexer(base_path=indexed_fixtures["bm25_path"])
        indexer.build([chunk_with_image], sparse, collection_name=collection)
        indexer.save()

        embedding = FakeEmbedding(dimension=16)
        vs = FakeVectorStore(collection_name=collection)
        vecs = embedding.embed([chunk_with_image.text])
        vs.upsert([
            VectorRecord(
                id=chunk_with_image.id,
                vector=vecs[0],
                text=chunk_with_image.text,
                metadata=chunk_with_image.metadata,
            )
        ])
        dense = DenseRetriever(embedding=embedding, vector_store=vs)
        sparse_ret = SparseRetriever(
            base_path=indexed_fixtures["bm25_path"],
            collection_name=collection,
        )
        hybrid = HybridSearch(dense_retriever=dense, sparse_retriever=sparse_ret)
        pipeline = RetrievalPipeline(
            query_processor=QueryProcessor(),
            hybrid_search=hybrid,
            reranker=RerankerOrchestrator(backend=NoneReranker()),
        )
        set_config_path("config/settings.yaml")  # 必须先于 set_pipeline，否则会清空 _pipeline
        set_pipeline(pipeline)
        from src.mcp_server.tools.query_knowledge_hub import set_images_base_path
        set_images_base_path(str(images_base))

        result = query_knowledge_hub(
            query="chart data science",
            collection_name=collection,
            top_k=5,
        )
        assert result.isError is False
        assert len(result.content) >= 1
        assert result.content[0].type == "text"
        has_image = any(c.type == "image" for c in result.content)
        assert has_image, f"Expected ImageContent, got: {[c.type for c in result.content]}"
        img_block = next(c for c in result.content if c.type == "image")
        assert img_block.mimeType == "image/png"
        decoded = base64.b64decode(img_block.data)
        assert decoded == img_bytes


class TestMCPServerE6:
    """E6 验收：错误处理与协议合规"""

    def test_query_knowledge_hub_empty_query_returns_error(self) -> None:
        """空 query 返回 isError=True，errorType=INVALID_PARAMS"""
        from src.mcp_server.tools.query_knowledge_hub import execute_query_knowledge_hub

        result = execute_query_knowledge_hub({"query": ""})
        assert result["isError"] is True
        assert len(result["content"]) >= 1
        assert "query" in result["content"][0]["text"].lower() or "参数" in result["content"][0]["text"]
        sc = result["structuredContent"]
        assert sc.get("errorType") == "INVALID_PARAMS"
        assert sc.get("errorCode") == -32602
        assert "message" in sc

    def test_get_document_summary_empty_doc_id_returns_error(self) -> None:
        """空 doc_id 返回 isError=True，errorType=INVALID_PARAMS"""
        from src.mcp_server.tools.get_document_summary import execute_get_document_summary

        result = execute_get_document_summary({"doc_id": ""})
        assert result["isError"] is True
        assert len(result["content"]) >= 1
        assert "doc_id" in result["content"][0]["text"].lower() or "参数" in result["content"][0]["text"]
        sc = result["structuredContent"]
        assert sc.get("errorType") == "INVALID_PARAMS"
        assert sc.get("errorCode") == -32602

    def test_get_document_summary_nonexistent_doc_returns_error(self) -> None:
        """不存在的 doc_id 返回 isError=True，errorType=RESOURCE_NOT_FOUND"""
        from src.mcp_server.tools.get_document_summary import (
            execute_get_document_summary,
            set_bm25_base_path,
        )
        import tempfile
        from pathlib import Path

        # 使用空 BM25 目录，确保 doc 不存在
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "bm25" / "any_coll"
            base.mkdir(parents=True)
            (base / "index.json").write_text('{"chunk_metadata":{}}', encoding="utf-8")
            set_bm25_base_path(str(Path(tmp) / "bm25"))

            result = execute_get_document_summary({"doc_id": "nonexistent_id_xyz"})
            assert result["isError"] is True
            assert "不存在" in result["content"][0]["text"] or "not found" in result["content"][0]["text"].lower()
            sc = result["structuredContent"]
            assert sc.get("errorType") == "RESOURCE_NOT_FOUND"
            assert sc.get("errorCode") == -32001

    def test_error_response_has_unified_format(self) -> None:
        """错误响应统一包含 content[0].text、structuredContent.errorCode/errorType/message"""
        from src.mcp_server.tools.get_document_summary import execute_get_document_summary

        result = execute_get_document_summary({"doc_id": ""})
        assert result["isError"] is True
        assert "content" in result
        assert len(result["content"]) >= 1
        assert result["content"][0]["type"] == "text"
        assert "text" in result["content"][0]
        sc = result["structuredContent"]
        assert "errorCode" in sc
        assert "errorType" in sc
        assert "message" in sc
