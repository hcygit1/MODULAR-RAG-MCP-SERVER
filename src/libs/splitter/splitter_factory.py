"""
Splitter 工厂模块

根据配置创建对应的 Splitter 实现实例。
支持通过配置文件切换不同的切分策略，无需修改代码。
"""
from typing import Optional

from src.core.settings import Settings
from src.libs.splitter.base_splitter import BaseSplitter


class SplitterFactory:
    """
    Splitter 工厂类
    
    根据配置动态创建对应的 Splitter 实现。
    使用工厂模式实现配置驱动的组件选择。
    """
    
    @staticmethod
    def create(settings: Settings, strategy: Optional[str] = None) -> BaseSplitter:
        """
        根据配置创建 Splitter 实例
        
        Args:
            settings: 配置对象，包含 Ingestion 配置信息
            strategy: 切分策略（可选），如果不提供则从配置中读取
                     支持的策略: "recursive", "semantic", "fixed"
            
        Returns:
            BaseSplitter: Splitter 实例
            
        Raises:
            ValueError: 当策略不支持或配置不完整时
            NotImplementedError: 当策略的实现在 B7 阶段尚未完成时
        """
        # 如果没有指定策略，默认使用 recursive
        # 注意：当前 IngestionConfig 中没有 splitter_strategy 字段
        # 这里先使用默认策略，后续可以在配置中添加
        if strategy is None:
            strategy = "recursive"  # 默认策略
        
        strategy = strategy.lower()
        ingestion_config = settings.ingestion
        
        if strategy == "recursive":
            # B7.5 阶段实现
            raise NotImplementedError(
                "Recursive Splitter 实现将在 B7.5 阶段完成。"
                "请先使用其他策略或等待实现。"
            )
        elif strategy == "semantic":
            # 未来实现
            raise NotImplementedError(
                "Semantic Splitter 实现尚未完成。"
                "请先使用其他策略或等待实现。"
            )
        elif strategy == "fixed":
            # 未来实现
            raise NotImplementedError(
                "Fixed Splitter 实现尚未完成。"
                "请先使用其他策略或等待实现。"
            )
        else:
            raise ValueError(
                f"不支持的 Splitter 策略: {strategy}。"
                f"支持的策略: recursive, semantic, fixed"
            )
    
    @staticmethod
    def create_fake(
        strategy: str = "fake",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> BaseSplitter:
        """
        创建 Fake Splitter 实例（用于测试）
        
        Args:
            strategy: 策略名称
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            BaseSplitter: Fake Splitter 实例
        """
        from src.libs.splitter.fake_splitter import FakeSplitter
        return FakeSplitter(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
