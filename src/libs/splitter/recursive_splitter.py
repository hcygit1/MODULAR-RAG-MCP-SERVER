"""
Recursive Splitter 实现

封装 LangChain 的 RecursiveCharacterTextSplitter，作为默认切分器。
能够正确处理 Markdown 结构（标题/代码块不被打断）。
"""
from typing import List, Optional, Any

from src.libs.splitter.base_splitter import BaseSplitter
from src.core.settings import IngestionConfig

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    # 如果 LangChain 未安装，提供一个占位实现
    RecursiveCharacterTextSplitter = None


class RecursiveSplitter(BaseSplitter):
    """
    Recursive Splitter 实现
    
    使用 LangChain 的 RecursiveCharacterTextSplitter 进行文本切分。
    能够智能识别文本结构（如 Markdown、代码块等），避免在不当位置切分。
    """
    
    def __init__(self, config: IngestionConfig):
        """
        初始化 Recursive Splitter
        
        Args:
            config: Ingestion 配置对象，包含 chunk_size 和 chunk_overlap
        """
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError(
                "LangChain 未安装。请安装 langchain: pip install langchain"
            )
        
        self._config = config
        self._strategy = "recursive"
        self._chunk_size = config.chunk_size
        self._chunk_overlap = config.chunk_overlap
        
        # 创建 LangChain RecursiveCharacterTextSplitter 实例
        # 默认分隔符优先级：\n\n, \n, " ", ""（按优先级尝试切分）
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,  # 使用字符数作为长度函数
            separators=[
                "\n\n",  # 段落分隔符（最高优先级）
                "\n",    # 换行符
                ". ",    # 句子分隔符
                " ",     # 单词分隔符
                ""       # 字符分隔符（最低优先级，作为最后手段）
            ]
        )
    
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None
    ) -> List[str]:
        """
        将文本切分为多个片段（chunks）
        
        使用 LangChain 的 RecursiveCharacterTextSplitter 进行智能切分。
        能够识别 Markdown 结构（标题、代码块等），避免在不当位置切分。
        
        Args:
            text: 输入文本，需要被切分的完整文本
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
        
        Returns:
            List[str]: 切分后的文本片段列表
        
        Raises:
            ValueError: 当输入文本为空或无效时
            RuntimeError: 当切分过程失败时
        """
        if not text:
            raise ValueError("输入文本不能为空")
        
        if not isinstance(text, str):
            raise ValueError(f"输入文本必须是字符串类型，得到: {type(text)}")
        
        try:
            # 使用 LangChain RecursiveCharacterTextSplitter 进行切分
            chunks = self._splitter.split_text(text)
            return chunks
        except Exception as e:
            raise RuntimeError(
                f"Recursive Splitter 切分失败 (strategy={self._strategy}): {str(e)}"
            ) from e
    
    def get_strategy(self) -> str:
        """获取策略名称"""
        return self._strategy
    
    def get_chunk_size(self) -> int:
        """获取块大小"""
        return self._chunk_size
    
    def get_chunk_overlap(self) -> int:
        """获取块重叠大小"""
        return self._chunk_overlap
