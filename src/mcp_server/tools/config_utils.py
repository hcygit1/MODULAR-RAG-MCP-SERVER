"""
MCP 工具统一配置获取

所有 MCP 工具从此模块读取配置路径与 settings，支持 MODULAR_RAG_CONFIG_PATH 环境变量。
测试可通过 set_config_path 覆盖配置路径。
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.core.settings import Settings

# 测试注入用：非 None 时优先于环境变量
_config_path_override: Optional[str] = None


def set_config_path(path: str) -> None:
    """设置配置路径（测试用）。传入后 get_config_path 返回此路径。"""
    global _config_path_override
    _config_path_override = path


def get_config_path() -> str:
    """返回配置路径，支持测试覆盖与 MODULAR_RAG_CONFIG_PATH 环境变量。"""
    if _config_path_override is not None:
        return _config_path_override
    return os.environ.get("MODULAR_RAG_CONFIG_PATH", "config/settings.yaml")


def load_mcp_settings() -> "Settings":
    """加载 MCP 工具需要的 settings，所有工具共用。"""
    from src.core.settings import load_settings

    return load_settings(get_config_path())
