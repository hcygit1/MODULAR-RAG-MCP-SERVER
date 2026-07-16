"""
E2E：MCP Client 侧调用模拟

以子进程启动 MCP Server，模拟 tools/list + tools/call，完整走通 query_knowledge_hub 并验证返回 citations。
验收标准：完整走通 query_knowledge_hub 并返回 citations。

每个测试动态生成 local embedding + SQLite 配置，保证离线且不读写业务数据库。
"""
import json
import subprocess
import sys
import threading
import time

import pytest


def _start_server_subprocess(
    e2e_runtime,
) -> subprocess.Popen:
    """启动 MCP Server 子进程，返回 Popen 实例。"""
    return subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
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


def _send_until_response(
    proc: subprocess.Popen,
    payload: str,
    response_id: int,
    timeout: float,
) -> tuple[str, str]:
    """保持 stdin 打开，读取目标响应后再关闭并回收子进程。"""
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    reader_errors: list[Exception] = []
    response_received = threading.Event()

    def read_stdout() -> None:
        try:
            for line in proc.stdout:
                stdout_lines.append(line)
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if response.get("id") == response_id:
                    response_received.set()
        except Exception as exc:
            reader_errors.append(exc)

    def read_stderr() -> None:
        try:
            stderr_lines.extend(proc.stderr)
        except Exception as exc:
            reader_errors.append(exc)

    stdout_reader = threading.Thread(target=read_stdout, daemon=True)
    stderr_reader = threading.Thread(target=read_stderr, daemon=True)
    stdout_reader.start()
    stderr_reader.start()

    deadline = time.monotonic() + timeout
    failure_reason: str | None = None
    try:
        proc.stdin.write(payload)
        proc.stdin.flush()

        while not response_received.is_set():
            if reader_errors:
                failure_reason = f"读取 MCP 子进程输出失败: {reader_errors[0]}"
                break
            if proc.poll() is not None:
                failure_reason = (
                    f"MCP Server 在响应 id={response_id} 前退出，"
                    f"returncode={proc.returncode}"
                )
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure_reason = f"等待 MCP 响应 id={response_id} 超时（{timeout} 秒）"
                break
            response_received.wait(min(0.05, remaining))

        if failure_reason is None:
            proc.stdin.close()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure_reason = f"MCP 响应后等待进程退出超时（{timeout} 秒）"
            else:
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    failure_reason = (
                        f"MCP 响应 id={response_id} 后未在 {timeout} 秒内退出"
                    )
    except (BrokenPipeError, OSError) as exc:
        failure_reason = f"与 MCP 子进程通信失败: {exc}"
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError as exc:
                if failure_reason is None:
                    failure_reason = f"关闭 MCP stdin 失败: {exc}"
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

        stdout_reader.join(timeout=1)
        stderr_reader.join(timeout=1)
        readers = (
            (stdout_reader, proc.stdout),
            (stderr_reader, proc.stderr),
        )
        for reader, stream in readers:
            if reader.is_alive() and not stream.closed:
                stream.close()
                reader.join(timeout=1)
        for reader, stream in readers:
            if not stream.closed:
                stream.close()
            if reader.is_alive() and failure_reason is None:
                failure_reason = f"MCP 输出 reader 未能退出: {reader.name}"
        if reader_errors and failure_reason is None:
            failure_reason = f"读取 MCP 子进程输出失败: {reader_errors[0]}"

    stdout_data = "".join(stdout_lines)
    stderr_data = "".join(stderr_lines)
    if failure_reason is not None:
        raise AssertionError(
            f"{failure_reason}\nstdout:\n{stdout_data}\nstderr:\n{stderr_data}"
        )
    return stdout_data, stderr_data


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
        _send_until_response(
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

    with pytest.raises(AssertionError, match="超时"):
        _send_until_response(
            proc,
            "",
            response_id=99,
            timeout=0.05,
        )

    assert proc.poll() is not None
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
    assert proc.stderr is not None and proc.stderr.closed


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
        stdout_data, stderr_data = _send_until_response(
            proc,
            payload,
            response_id=1,
            timeout=10,
        )
        assert proc.returncode == 0, f"Server exited with {proc.returncode}, stderr: {stderr_data}"

        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        assert len(lines) >= 2, f"Expected at least 2 response lines, got: {stdout_data}"
        responses = [json.loads(line) for line in lines]
        tools_resp = next(
            (response for response in responses if response.get("id") == 1),
            None,
        )
        assert tools_resp is not None, f"未找到 tools/list id=1 响应: {stdout_data}"
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
        stdout_data, stderr_data = _send_until_response(
            proc,
            payload,
            response_id=2,
            timeout=30,
        )
        assert proc.returncode == 0, f"Server exited with {proc.returncode}, stderr: {stderr_data}"

        lines = [line.strip() for line in stdout_data.strip().split("\n") if line.strip()]
        assert len(lines) >= 3, f"Expected at least 3 response lines, got {len(lines)}: {stdout_data}"

        responses = [json.loads(line) for line in lines]
        call_resp = next(
            (response for response in responses if response.get("id") == 2),
            None,
        )
        assert call_resp is not None, f"未找到 tools/call id=2 响应: {stdout_data}"
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
