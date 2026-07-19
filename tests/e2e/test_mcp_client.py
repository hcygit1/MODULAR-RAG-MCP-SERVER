"""
E2E：MCP Client 侧调用模拟

以子进程启动 MCP Server，模拟 tools/list + tools/call，完整走通 query_knowledge_hub 并验证返回 citations。
验收标准：完整走通 query_knowledge_hub 并返回 citations。

每个测试动态生成 local embedding + SQLite 配置，保证离线且不读写业务数据库。
"""
import json
import subprocess
import sys
import time

import pytest

from tests.support import mcp_process
from tests.support.mcp_process import jsonrpc_response, send_until_response


def _start_server_subprocess(
    e2e_runtime,
) -> subprocess.Popen:
    """启动 MCP Server 子进程，返回 Popen 实例。"""
    return subprocess.Popen(
        [sys.executable, "-m", "modular_rag_mcp_server.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=e2e_runtime["workdir"],
        env=e2e_runtime["env"],
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


def test_send_until_response_closes_process_streams() -> None:
    script = (
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], "
        "'result': {}}), flush=True)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        send_until_response(
            proc,
            json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"}) + "\n",
            response_id=7,
            timeout=2,
        )

        assert proc.poll() is not None
        assert proc.stdin is not None and proc.stdin.closed
        assert proc.stdout is not None and proc.stdout.closed
        assert proc.stderr is not None and proc.stderr.closed
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def test_send_until_response_kills_and_reaps_on_timeout() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    with pytest.raises(AssertionError, match="Timed out"):
        send_until_response(
            proc,
            "",
            response_id=99,
            timeout=0.05,
        )

    assert proc.poll() is not None
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
    assert proc.stderr is not None and proc.stderr.closed


def test_send_until_response_accepts_response_before_process_exit(monkeypatch) -> None:
    script = (
        "import json\n"
        "print(json.dumps({'jsonrpc': '2.0', 'id': 7, 'result': {}}), flush=True)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    original_loads = mcp_process.json.loads

    def delayed_loads(line):
        time.sleep(0.1)
        return original_loads(line)

    monkeypatch.setattr(mcp_process.json, "loads", delayed_loads)

    stdout_data, _ = send_until_response(
        proc,
        "",
        response_id=7,
        timeout=2,
    )

    assert jsonrpc_response(stdout_data, 7)["result"] == {}


def test_send_until_response_times_out_when_child_does_not_read_payload() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    started_at = time.monotonic()

    with pytest.raises(AssertionError, match="Timed out"):
        send_until_response(
            proc,
            "x" * 1_000_000,
            response_id=99,
            timeout=0.05,
        )

    assert time.monotonic() - started_at < 1.5
    assert proc.poll() is not None


class TestMCPClientE2E:
    """G1 验收：E2E MCP Client 侧调用模拟"""

    def test_tools_list_returns_tools(self, e2e_runtime) -> None:
        """tools/list 能返回已注册 tools 列表"""
        proc = _start_server_subprocess(e2e_runtime)
        payload = _mcp_handshake() + json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        }) + "\n"
        stdout_data, stderr_data = send_until_response(
            proc,
            payload,
            response_id=1,
            timeout=10,
        )
        assert proc.returncode == 0, f"Server exited with {proc.returncode}, stderr: {stderr_data}"

        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        assert len(lines) >= 2, f"Expected at least 2 response lines, got: {stdout_data}"
        tools_resp = jsonrpc_response(stdout_data, 1)
        assert "result" in tools_resp
        assert "tools" in tools_resp["result"]
        tools = tools_resp["result"]["tools"]
        names = [t["name"] for t in tools]
        assert "query_knowledge_hub" in names
        assert "list_collections" in names

    def test_query_knowledge_hub_returns_citations(self, ingested_e2e_runtime) -> None:
        """tools/call query_knowledge_hub 完整走通并返回 citations"""
        proc = _start_server_subprocess(ingested_e2e_runtime)
        handshake = _mcp_handshake()
        tools_list_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        }
        query_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {
                    "query": "test document",
                    "collection_name": "test_collection",
                    "top_k": 5,
                },
            },
        }
        payload = handshake + json.dumps(tools_list_req) + "\n" + json.dumps(query_req) + "\n"
        stdout_data, stderr_data = send_until_response(
            proc,
            payload,
            response_id=2,
            timeout=30,
        )
        assert proc.returncode == 0, f"Server exited with {proc.returncode}, stderr: {stderr_data}"

        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        assert len(lines) >= 3, f"Expected at least 3 response lines, got {len(lines)}: {stdout_data}"

        call_resp = jsonrpc_response(stdout_data, 2)
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
