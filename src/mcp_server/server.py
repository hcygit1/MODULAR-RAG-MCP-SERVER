"""
MCP Server 入口 (FastMCP + Stdio Transport)

使用官方 MCP SDK (FastMCP)，遵循协议约束：stdout 仅输出 MCP 消息，日志输出到 stderr。
"""
from mcp.server.fastmcp import FastMCP

from src.mcp_server.tools.get_document_summary import get_document_summary
from src.mcp_server.tools.list_collections import list_collections
from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub

mcp = FastMCP("modular-rag-mcp-server")

mcp.tool()(query_knowledge_hub)
mcp.tool()(list_collections)
mcp.tool()(get_document_summary)


def run_server() -> None:
    """启动 MCP Server（stdio transport）。"""
    mcp.run(transport="stdio")


def main() -> None:
    """命令行入口，供 python -m src.mcp_server.server 调用。"""
    run_server()


if __name__ == "__main__":
    main()
