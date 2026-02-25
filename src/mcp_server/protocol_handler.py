"""
MCP Protocol Handler

封装 JSON-RPC 2.0 协议解析，处理 initialize、tools/list、tools/call 三类核心方法。
遵循 MCP 规范，错误码符合 JSON-RPC 2.0。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypedDict

# JSON-RPC 2.0 错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


class ProtocolError(Exception):
    """协议层错误，用于向调用方传递 JSON-RPC 错误码与消息。"""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class ToolDefinition(TypedDict, total=False):
    """MCP 工具定义 schema"""

    name: str
    description: str
    inputSchema: Dict[str, Any]


ToolExecutor = Callable[[Dict[str, Any]], Any]


class ProtocolHandler:
    """
    MCP 协议处理器。

    负责 initialize、tools/list、tools/call 的解析与路由，
    将异常转换为 JSON-RPC 规范错误。
    """

    def __init__(self) -> None:
        """初始化协议处理器，工具注册表为空。"""
        self._tools: Dict[str, tuple[ToolDefinition, ToolExecutor]] = {}

    def register_tool(self, definition: ToolDefinition, executor: ToolExecutor) -> None:
        """
        注册一个工具。

        Args:
            definition: 工具 schema（name, description, inputSchema）
            executor: 执行函数，接收 arguments 字典，返回任意可序列化结果
        """
        name = definition.get("name", "")
        if not name:
            raise ValueError("Tool definition must have 'name'")
        self._tools[name] = (definition, executor)

    def handle_initialize(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理 initialize 请求，返回 serverInfo 和 capabilities。

        Args:
            params: 客户端传入的初始化参数（可选）

        Returns:
            MCP initialize result，含 protocolVersion、serverInfo、capabilities
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

    def handle_tools_list(self) -> Dict[str, Any]:
        """
        返回已注册工具的 schema 列表。

        Returns:
            MCP tools/list result，含 tools 数组（name, description, inputSchema）
        """
        tools: List[ToolDefinition] = []
        for defn, _ in self._tools.values():
            tools.append(defn)
        return {"tools": tools}

    def handle_tools_call(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        根据 name 路由到对应工具执行，捕获异常并转换为规范结果。

        Args:
            name: 工具名称
            arguments: 调用参数，默认为空字典

        Returns:
            MCP tools/call result：{ content: [...], isError: bool }
            成功时 content 为 TextContent 列表；失败时 isError=True，content 含错误信息。

        Raises:
            不直接抛出，将异常信息封装进返回的 content 中（isError=True）
        """
        args = arguments if arguments is not None else {}

        if name not in self._tools:
            return {
                "content": [
                    {"type": "text", "text": f"Unknown tool: {name}"},
                ],
                "isError": True,
            }

        _, executor = self._tools[name]
        try:
            result = executor(args)
            return _normalize_tool_result(result)
        except Exception as e:
            return {
                "content": [
                    {"type": "text", "text": str(e)},
                ],
                "isError": True,
            }

    def dispatch(self, method: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        根据 JSON-RPC method 分发到具体处理逻辑。

        Args:
            method: JSON-RPC method 名称（initialize、tools/list、tools/call）
            params: 请求参数

        Returns:
            成功时返回 result 字典

        Raises:
            ProtocolError: params 无效时抛出（code=-32602）；method 不存在时不抛，由调用方处理
        """
        if method == "initialize":
            return self.handle_initialize(params)

        if method == "tools/list":
            return self.handle_tools_list()

        if method == "tools/call":
            if not isinstance(params, dict):
                raise ProtocolError(INVALID_PARAMS, "params must be an object")
            name = params.get("name")
            if name is None or not isinstance(name, str) or not name.strip():
                raise ProtocolError(INVALID_PARAMS, "params.name is required and must be a non-empty string")
            arguments = params.get("arguments")
            if arguments is not None and not isinstance(arguments, dict):
                raise ProtocolError(INVALID_PARAMS, "params.arguments must be an object")
            return self.handle_tools_call(name, arguments)

        raise ProtocolError(METHOD_NOT_FOUND, f"Method not found: {method}")


def _normalize_tool_result(raw: Any) -> Dict[str, Any]:
    """
    将工具返回值规范化为 MCP content 格式。

    支持：
    - str -> TextContent
    - list[dict] (已是 MCP content) -> 直接使用
    - 其他 -> 转 str 后包装为 TextContent
    """
    if isinstance(raw, str):
        return {
            "content": [{"type": "text", "text": raw}],
            "isError": False,
        }
    if isinstance(raw, dict) and "content" in raw:
        return {"content": raw["content"], "isError": raw.get("isError", False)}
    return {
        "content": [{"type": "text", "text": str(raw)}],
        "isError": False,
    }
