"""
MCP Server 入口 (Stdio Transport)

遵循 MCP 协议约束：stdout 仅输出 MCP 消息（JSON-RPC），日志输出到 stderr。
"""
import json
import sys
from typing import Any, Dict, Optional

# 使用 stderr 输出日志，避免污染 stdout 的 MCP 消息
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


def _handle_initialize(_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    处理 initialize 请求，返回 serverInfo 和 capabilities。

    Args:
        _params: 客户端传入的初始化参数（可选）

    Returns:
        result 对象，含 protocolVersion、serverInfo、capabilities
    """
    return {
        "protocolVersion": "2024-11-05",
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
        },
        "capabilities": {
            "tools": {},
        },
    }


def _dispatch(method: str, params: Optional[Dict[str, Any]], request_id: Any) -> Optional[Dict[str, Any]]:
    """
    根据 method 分发到具体处理逻辑。

    Args:
        method: JSON-RPC method 名称
        params: 请求参数
        request_id: 请求 id，用于响应

    Returns:
        成功时返回 result 字典；失败时返回 None（将由调用方构造 error 响应）
    """
    if method == "initialize":
        return _handle_initialize(params)
    return None


def _write_response(response: Dict[str, Any]) -> None:
    """将 JSON-RPC 响应写入 stdout（MCP 消息通道）。"""
    line = json.dumps(response, ensure_ascii=False) + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()


def _write_error(request_id: Any, code: int, message: str) -> None:
    """写入 JSON-RPC 错误响应到 stdout。"""
    _write_response({
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    })


def _process_message(line: str) -> None:
    """
    处理单条 JSON-RPC 消息。

    Args:
        line: 一行 JSON 字符串（不含末尾换行）
    """
    line = line.strip()
    if not line:
        return

    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON: %s", e)
        _write_error(None, -32700, "Parse error")
        return

    if not isinstance(req, dict):
        _write_error(None, -32600, "Invalid Request")
        return

    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params")

    if method is None:
        _write_error(req_id, -32600, "Invalid Request: missing method")
        return

    # Notification 无 id，不返回响应
    if req_id is None:
        logger.debug("Notification ignored: method=%s", method)
        return

    result = _dispatch(method, params, req_id)
    if result is not None:
        _write_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        })
    else:
        _write_error(req_id, -32601, f"Method not found: {method}")


def run_server() -> None:
    """
    启动 MCP Server，从 stdin 读取 JSON-RPC 消息，向 stdout 写入响应。

    日志全部输出到 stderr，保证 stdout 仅含 MCP 协议消息。
    """
    logger.info("MCP Server starting (stdio transport)")
    try:
        for line in sys.stdin:
            _process_message(line)
    except Exception as e:
        logger.exception("Server error: %s", e)
        raise
    finally:
        logger.info("MCP Server stopped")


def main() -> None:
    """命令行入口，供 python -m src.mcp_server.server 调用。"""
    run_server()


if __name__ == "__main__":
    main()
