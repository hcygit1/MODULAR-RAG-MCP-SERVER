"""
MCP Server 集成测试

验证 Server 能完成 initialize；stderr 有日志，stdout 仅含 MCP 消息（不污染）。
以子进程方式启动 server，发送 initialize 请求，断言响应与日志分离。
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
