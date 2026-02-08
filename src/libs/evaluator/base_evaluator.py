"""
Evaluator 抽象接口模块

定义统一的评估接口，所有 Evaluator 实现（Custom、Ragas、DeepEval 等）
都必须遵循此接口。
"""
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseEvaluator(ABC):
    """
    Evaluator 抽象基类
    
    定义所有评估实现必须遵循的统一接口。
    无论底层使用自定义指标、Ragas 还是 DeepEval，
    上层代码都可以通过统一的接口调用。
    """
    
    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str]
    ) -> Dict[str, float]:
        """
        评估检索结果的质量
        
        Args:
            query: 查询文本
            retrieved_ids: 检索返回的文档 ID 列表（按相关性排序）
            golden_ids: 标准答案的文档 ID 列表（ground truth）
        
        Returns:
            Dict[str, float]: 评估指标字典
                             - 键：指标名称，例如 "hit_rate", "mrr", "ndcg"
                             - 值：指标分数（通常范围 0.0-1.0）
                             - 示例: {"hit_rate": 0.8, "mrr": 0.6}
        
        Raises:
            ValueError: 当输入参数无效时（例如空列表）
        """
        pass
    
    @abstractmethod
    def get_backend(self) -> str:
        """
        获取当前使用的 backend 名称
        
        Returns:
            str: backend 名称，例如 "custom", "ragas", "deepeval"
        """
        pass
