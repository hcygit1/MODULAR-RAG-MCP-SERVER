"""
MCP Server 入口 (FastMCP + Stdio Transport)

使用官方 MCP SDK (FastMCP)，遵循协议约束：stdout 仅输出 MCP 消息，日志输出到 stderr。
"""
from mcp.server.fastmcp import FastMCP

from src.mcp_server.tools.list_collections import list_collections
from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub
from src.mcp_server.tools.ingest_document_mineru import ingest_document_mineru

mcp = FastMCP("modular-rag-mcp-server")

# 工具描述需明确「何时调用」，便于 Cursor/Agent 识别用户意图并决策
mcp.tool(description=(
    "当用户询问知识库内容、文档相关问题、或需要检索资料时调用。"
    "在 RAG 知识库中语义检索，返回相关文档片段（Markdown）和引用。"
    "参数: query(必填), collection_name(可选), top_k(可选，默认10)。"
))(query_knowledge_hub)
mcp.tool(description=(
    "当用户想知道有哪些知识库集合、或需要选择/确认 collection 时调用。"
    "列出已入库的集合名称。"
))(list_collections)
mcp.tool(description=(
    "当用户要把 PDF 加入知识库、或要求用 MinerU 精细解析/入库时调用。"
    "使用 MinerU 云端 API 解析 PDF 并写入向量库，适用于表格多、公式多、扫描件等复杂排版。"
    "参数: file_path(必填), collection_name(可选)。"
))(ingest_document_mineru)


def run_server() -> None:
    """启动 MCP Server（stdio transport）。"""
    mcp.run(transport="stdio")


def main() -> None:
    """命令行入口，供 python -m src.mcp_server.server 调用。"""
    run_server()


if __name__ == "__main__":
    main()
