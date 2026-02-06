"""
日志模块

提供统一的日志接口，输出到 stderr（避免污染 stdout 的 MCP 消息）。
"""
import logging
import sys
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取 logger 实例
    
    Args:
        name: logger 名称，默认为 None（使用调用模块名）
        
    Returns:
        logging.Logger: 配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    
    # 如果 logger 还没有 handler，添加一个
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger
