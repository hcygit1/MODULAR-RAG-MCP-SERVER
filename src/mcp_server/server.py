"""
MCP Server 入口 (Stdio Transport)

遵循 MCP 协议约束：stdout 仅输出 MCP 消息（JSON-RPC），日志输出到 stderr。
"""
import json
import sys
from typing import Any, Dict, Optional

# 使用 stderr 输出日志，避免污染 stdout 的 MCP 消息
import logging

from src.mcp_server.protocol_handler import ProtocolError, ProtocolHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# 模块级 Handler，run_server 使用
_handler: Optional[ProtocolHandler] = None


def _get_handler() -> ProtocolHandler:
    """获取或创建 ProtocolHandler 实例，并注册 query_knowledge_hub 工具。"""
    global _handler
    if _handler is None:
        from src.mcp_server.tools.get_document_summary import (
            GET_DOCUMENT_SUMMARY_DEFINITION,
            execute_get_document_summary,
        )
        from src.mcp_server.tools.list_collections import (
            LIST_COLLECTIONS_DEFINITION,
            execute_list_collections,
        )
        from src.mcp_server.tools.query_knowledge_hub import (
            QUERY_KNOWLEDGE_HUB_DEFINITION,
            execute_query_knowledge_hub,
        )

        _handler = ProtocolHandler()
        _handler.register_tool(QUERY_KNOWLEDGE_HUB_DEFINITION, execute_query_knowledge_hub)
        _handler.register_tool(LIST_COLLECTIONS_DEFINITION, execute_list_collections)
        _handler.register_tool(GET_DOCUMENT_SUMMARY_DEFINITION, execute_get_document_summary)
    return _handler


def set_handler(handler: ProtocolHandler) -> None:
    """测试注入用：替换默认 handler。"""
    global _handler
    _handler = handler


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

    handler = _get_handler()
    try:
        result = handler.dispatch(method, params)
        _write_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        })
    except ProtocolError as e:
        _write_error(req_id, e.code, e.message)


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
