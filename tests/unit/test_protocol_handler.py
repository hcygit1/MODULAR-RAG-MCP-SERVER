"""
Protocol Handler 单元测试

验证 initialize、tools/list、tools/call 的处理逻辑，
以及错误码符合 JSON-RPC 2.0 规范。
"""
import pytest

from src.mcp_server.protocol_handler import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    ProtocolError,
    ProtocolHandler,
    SERVER_NAME,
    SERVER_VERSION,
)


class TestProtocolHandlerInitialize:
    """initialize 请求"""

    def test_handle_initialize_returns_server_info(self) -> None:
        """返回正确的 serverInfo 和 capabilities"""
        handler = ProtocolHandler()
        result = handler.handle_initialize()
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == SERVER_NAME
        assert result["serverInfo"]["version"] == SERVER_VERSION
        assert "capabilities" in result
        assert "tools" in result["capabilities"]

    def test_handle_initialize_accepts_params(self) -> None:
        """接受可选的 params 参数"""
        handler = ProtocolHandler()
        result = handler.handle_initialize({"clientInfo": {"name": "test"}})
        assert result["serverInfo"]["name"] == SERVER_NAME


class TestProtocolHandlerToolsList:
    """tools/list 请求"""

    def test_handle_tools_list_empty_when_no_tools(self) -> None:
        """无注册工具时返回空列表"""
        handler = ProtocolHandler()
        result = handler.handle_tools_list()
        assert result == {"tools": []}

    def test_handle_tools_list_returns_registered_schemas(self) -> None:
        """返回已注册工具的 schema"""
        handler = ProtocolHandler()
        handler.register_tool(
            {
                "name": "echo",
                "description": "Echo input",
                "inputSchema": {
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                },
            },
            lambda args: args.get("msg", ""),
        )
        result = handler.handle_tools_list()
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "echo"
        assert result["tools"][0]["description"] == "Echo input"
        assert result["tools"][0]["inputSchema"]["type"] == "object"


class TestProtocolHandlerToolsCall:
    """tools/call 请求"""

    def test_handle_tools_call_unknown_tool_returns_error_content(self) -> None:
        """未知工具返回 isError=True 的 content"""
        handler = ProtocolHandler()
        result = handler.handle_tools_call("unknown_tool", {})
        assert result["isError"] is True
        assert "content" in result
        assert len(result["content"]) >= 1
        assert "Unknown tool" in result["content"][0]["text"]

    def test_handle_tools_call_invokes_registered_tool(self) -> None:
        """正确路由并执行已注册工具"""
        handler = ProtocolHandler()

        def echo_executor(args: dict) -> str:
            return args.get("msg", "no msg")

        handler.register_tool(
            {"name": "echo", "description": "Echo", "inputSchema": {"type": "object"}},
            echo_executor,
        )
        result = handler.handle_tools_call("echo", {"msg": "hello"})
        assert result["isError"] is False
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "hello"

    def test_handle_tools_call_defaults_empty_arguments(self) -> None:
        """arguments 为 None 时视为空字典"""
        handler = ProtocolHandler()
        handler.register_tool(
            {"name": "noop", "description": "No-op", "inputSchema": {"type": "object"}},
            lambda args: "ok",
        )
        result = handler.handle_tools_call("noop", None)
        assert result["isError"] is False
        assert result["content"][0]["text"] == "ok"

    def test_handle_tools_call_catches_exception(self) -> None:
        """工具执行异常时返回 isError=True"""
        handler = ProtocolHandler()
        handler.register_tool(
            {"name": "fail", "description": "Fails", "inputSchema": {"type": "object"}},
            lambda _: (_ for _ in ()).throw(ValueError("intentional")),
        )
        result = handler.handle_tools_call("fail", {})
        assert result["isError"] is True
        assert "intentional" in result["content"][0]["text"]


class TestProtocolHandlerDispatch:
    """dispatch 分发逻辑"""

    def test_dispatch_initialize(self) -> None:
        """dispatch(initialize) 返回 serverInfo"""
        handler = ProtocolHandler()
        result = handler.dispatch("initialize", None)
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == SERVER_NAME

    def test_dispatch_tools_list(self) -> None:
        """dispatch(tools/list) 返回 tools 数组"""
        handler = ProtocolHandler()
        result = handler.dispatch("tools/list", None)
        assert "tools" in result
        assert result["tools"] == []

    def test_dispatch_tools_call_valid(self) -> None:
        """dispatch(tools/call) 有效参数时执行工具"""
        handler = ProtocolHandler()
        handler.register_tool(
            {"name": "x", "description": "", "inputSchema": {}},
            lambda a: a.get("k", ""),
        )
        result = handler.dispatch("tools/call", {"name": "x", "arguments": {"k": "v"}})
        assert result["content"][0]["text"] == "v"

    def test_dispatch_tools_call_missing_name_raises(self) -> None:
        """tools/call 缺 name 时抛出 ProtocolError -32602"""
        handler = ProtocolHandler()
        with pytest.raises(ProtocolError) as exc_info:
            handler.dispatch("tools/call", {})
        assert exc_info.value.code == INVALID_PARAMS
        assert "name" in exc_info.value.message.lower()

    def test_dispatch_tools_call_empty_name_raises(self) -> None:
        """tools/call name 为空字符串时抛出 -32602"""
        handler = ProtocolHandler()
        with pytest.raises(ProtocolError) as exc_info:
            handler.dispatch("tools/call", {"name": "", "arguments": {}})
        assert exc_info.value.code == INVALID_PARAMS

    def test_dispatch_tools_call_params_not_object_raises(self) -> None:
        """tools/call params 非对象时抛出 -32602"""
        handler = ProtocolHandler()
        with pytest.raises(ProtocolError) as exc_info:
            handler.dispatch("tools/call", "invalid")
        assert exc_info.value.code == INVALID_PARAMS

    def test_dispatch_unknown_method_raises(self) -> None:
        """未知 method 抛出 -32601"""
        handler = ProtocolHandler()
        with pytest.raises(ProtocolError) as exc_info:
            handler.dispatch("unknown/method", None)
        assert exc_info.value.code == METHOD_NOT_FOUND
        assert "unknown" in exc_info.value.message.lower() or "method" in exc_info.value.message.lower()
