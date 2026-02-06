"""
Fake Splitter 实现（用于测试）

提供一个简单的 Fake Splitter 实现，用于测试工厂路由逻辑，
不进行真实的文本切分算法。
"""
from typing import List, Optional, Any

from src.libs.splitter.base_splitter import BaseSplitter


class FakeSplitter(BaseSplitter):
    """
    Fake Splitter 实现
    
    用于测试和开发阶段，返回简单的固定长度切分结果。
    实际切分算法将在 B7.5 阶段实现。
    """
    
    def __init__(
        self,
        strategy: str = "fake",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        初始化 Fake Splitter
        
        Args:
            strategy: 策略名称
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠大小（字符数）
        """
        self._strategy = strategy
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
    
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None
    ) -> List[str]:
        """
        简单的固定长度切分实现
        
        将文本按照 chunk_size 切分，相邻块之间有 chunk_overlap 的重叠。
        
        Args:
            text: 输入文本
            trace: 追踪上下文（可选，Fake 实现中不使用）
            
        Returns:
            List[str]: 切分后的文本片段列表
            
        Raises:
            ValueError: 当输入文本为空时
        """
        if not text:
            raise ValueError("输入文本不能为空")
        
        # 如果文本长度小于等于 chunk_size，直接返回
        if len(text) <= self._chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # 计算当前块的结束位置
            end = start + self._chunk_size
            
            # 提取当前块
            chunk = text[start:end]
            chunks.append(chunk)
            
            # 如果已经到达文本末尾，退出循环
            if end >= len(text):
                break
            
            # 计算下一个块的起始位置（考虑重叠）
            start = end - self._chunk_overlap
        
        return chunks
    
    def get_strategy(self) -> str:
        """获取策略名称"""
        return self._strategy
    
    def get_chunk_size(self) -> int:
        """获取块大小"""
        return self._chunk_size
    
    def get_chunk_overlap(self) -> int:
        """获取块重叠大小"""
        return self._chunk_overlap
