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
        assert result["serverInfo"]["version"] == "0.1.0"
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

        # stderr 应有日志（如 "MCP Server" 或 "INFO"）
        assert len(stderr_data) > 0, "stderr 应有日志输出"
        assert "MCP" in stderr_data or "INFO" in stderr_data or "Server" in stderr_data, (
            f"stderr 应有 server 日志，实际: {stderr_data[:200]}"
        )


def _send_request(proc: subprocess.Popen, request: dict) -> tuple[str, str]:
    """发送单条 JSON-RPC 请求，返回 (stdout, stderr)。"""
    req_line = json.dumps(request) + "\n"
    stdout_data, stderr_data = proc.communicate(input=req_line, timeout=5)
    return stdout_data, stderr_data


class TestMCPServerE15:
    """E1.5 验收：tools/list、tools/call 协议层"""

    def test_tools_list_returns_schema(self) -> None:
        """发送 tools/list 能返回 tools 数组（可为空）"""
        proc = _start_server_subprocess()
        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        stdout_data, _ = _send_request(proc, request)
        assert proc.returncode == 0
        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        assert len(lines) >= 1
        resp = json.loads(lines[0])
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
        stdout_data, _ = _send_request(proc, request)
        assert proc.returncode == 0
        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        resp = json.loads(lines[0])
        assert "result" in resp
        result = resp["result"]
        assert result.get("isError") is True
        assert "content" in result
        assert len(result["content"]) >= 1
        assert "Unknown tool" in result["content"][0].get("text", "")

    def test_tools_call_missing_name_returns_invalid_params(self) -> None:
        """tools/call 缺 name 时返回 JSON-RPC -32602"""
        proc = _start_server_subprocess()
        request = {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {}}
        stdout_data, _ = _send_request(proc, request)
        assert proc.returncode == 0
        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        resp = json.loads(lines[0])
        assert "error" in resp
        assert resp["error"]["code"] == -32602


class TestMCPServerE2:
    """E2 验收：query_knowledge_hub Tool"""

    def test_tools_list_includes_query_knowledge_hub(self) -> None:
        """tools/list 包含 query_knowledge_hub schema"""
        proc = _start_server_subprocess()
        request = {"jsonrpc": "2.0", "id": 10, "method": "tools/list"}
        stdout_data, _ = _send_request(proc, request)
        assert proc.returncode == 0
        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        resp = json.loads(lines[0])
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
        stdout_data, _ = _send_request(proc, request)
        assert proc.returncode == 0
        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        resp = json.loads(lines[0])
        names = [t["name"] for t in resp["result"]["tools"]]
        assert "list_collections" in names

    def test_query_knowledge_hub_returns_markdown_and_citations(
        self, retrieval_pipeline, indexed_fixtures
    ) -> None:
        """tools/call query_knowledge_hub 返回 content[0] 为 Markdown，structuredContent.citations 含 source/page/chunk_id/score"""
        from src.mcp_server.protocol_handler import ProtocolHandler
        from src.mcp_server.tools.query_knowledge_hub import (
            QUERY_KNOWLEDGE_HUB_DEFINITION,
            execute_query_knowledge_hub,
            set_pipeline,
        )

        set_pipeline(retrieval_pipeline)

        handler = ProtocolHandler()
        handler.register_tool(QUERY_KNOWLEDGE_HUB_DEFINITION, execute_query_knowledge_hub)

        result = handler.handle_tools_call(
            "query_knowledge_hub",
            {
                "query": "python data science",
                "collection_name": indexed_fixtures["collection_name"],
                "top_k": 5,
            },
        )

        assert result["isError"] is False
        assert "content" in result and len(result["content"]) >= 1
        assert result["content"][0]["type"] == "text"
        assert len(result["content"][0]["text"]) > 0
        # Markdown 应包含片段标题或来源
        text = result["content"][0]["text"]
        assert "片段" in text or "来源" in text or "Python" in text

        assert "structuredContent" in result
        assert "citations" in result["structuredContent"]
        citations = result["structuredContent"]["citations"]
        assert len(citations) >= 1
        for c in citations:
            assert "source" in c
            assert "chunk_id" in c
            assert "score" in c
