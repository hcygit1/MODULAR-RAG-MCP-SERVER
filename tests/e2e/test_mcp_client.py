"""
E2E：MCP Client 侧调用模拟

以子进程启动 MCP Server，模拟 tools/list + tools/call，完整走通 query_knowledge_hub 并验证返回 citations。
验收标准：完整走通 query_knowledge_hub 并返回 citations。

使用 config/settings.e2e.yaml（local embedding + chroma）保证离线可运行，
通过环境变量 MODULAR_RAG_CONFIG_PATH 传入。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
E2E_CONFIG = str(PROJECT_ROOT / "config" / "settings.e2e.yaml")


def _start_server_subprocess(
    cwd: Path | None = None,
    use_e2e_config: bool = True,
) -> subprocess.Popen:
    """启动 MCP Server 子进程，返回 Popen 实例。"""
    env = os.environ.copy()
    if use_e2e_config and Path(E2E_CONFIG).exists():
        env["MODULAR_RAG_CONFIG_PATH"] = E2E_CONFIG
    return subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd or PROJECT_ROOT),
        env=env,
    )


def _mcp_handshake() -> str:
    """MCP 握手：initialize + notifications/initialized。"""
    init_req = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "e2e-test-client", "version": "0.1.0"},
        },
    }
    notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    return json.dumps(init_req) + "\n" + json.dumps(notif) + "\n"


class TestMCPClientE2E:
    """G1 验收：E2E MCP Client 侧调用模拟"""

    def test_tools_list_returns_tools(self) -> None:
        """tools/list 能返回已注册 tools 列表"""
        proc = _start_server_subprocess()
        payload = _mcp_handshake() + json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        }) + "\n"
        stdout_data, stderr_data = proc.communicate(input=payload, timeout=10)
        assert proc.returncode == 0, f"Server exited with {proc.returncode}, stderr: {stderr_data}"

        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        assert len(lines) >= 2, f"Expected at least 2 response lines, got: {stdout_data}"
        tools_resp = json.loads(lines[-1])
        assert "result" in tools_resp
        assert "tools" in tools_resp["result"]
        tools = tools_resp["result"]["tools"]
        names = [t["name"] for t in tools]
        assert "query_knowledge_hub" in names
        assert "list_collections" in names

    def test_query_knowledge_hub_returns_citations(self) -> None:
        """tools/call query_knowledge_hub 完整走通并返回 citations"""
        proc = _start_server_subprocess()
        handshake = _mcp_handshake()
        tools_list_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        }
        # 使用与 BM25 索引中存在的词匹配的查询（report 集合含中文文档，用「系统」「数据」等）
        query_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {
                    "query": "氢能 系统 数据",
                    "collection_name": "report",
                    "top_k": 5,
                },
            },
        }
        payload = handshake + json.dumps(tools_list_req) + "\n" + json.dumps(query_req) + "\n"
        stdout_data, stderr_data = proc.communicate(input=payload, timeout=30)
        assert proc.returncode == 0, f"Server exited with {proc.returncode}, stderr: {stderr_data}"

        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        assert len(lines) >= 3, f"Expected at least 3 response lines, got {len(lines)}: {stdout_data}"

        # 最后一条为 tools/call 的响应
        call_resp = json.loads(lines[-1])
        assert call_resp.get("id") == 2
        assert "result" in call_resp
        result = call_resp["result"]

        assert result.get("isError") is False, (
            f"query_knowledge_hub 不应失败: {result.get('content', [])}"
        )
        assert "content" in result
        assert len(result["content"]) >= 1
        # Markdown 文本
        first_content = result["content"][0]
        assert first_content.get("type") == "text"
        assert len(first_content.get("text", "")) > 0

        # structuredContent.citations
        assert "structuredContent" in result
        structured = result["structuredContent"]
        assert "citations" in structured
        citations = structured["citations"]
        assert isinstance(citations, list)
        assert len(citations) >= 1, "应至少返回 1 条 citation"
        for c in citations:
            assert "source" in c
            assert "chunk_id" in c
            assert "score" in c
