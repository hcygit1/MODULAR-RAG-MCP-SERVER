"""
Custom Evaluator 实现

提供轻量级的自定义评估指标实现，包括 hit_rate 和 MRR。
不依赖外部评估库，适合快速回归测试和上线前 Sanity Check。
"""
from typing import List, Dict

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """
    Custom Evaluator 实现
    
    实现轻量级评估指标：
    - Hit Rate（命中率）：检索结果中包含标准答案的比例
    - MRR（Mean Reciprocal Rank，平均倒数排名）：第一个标准答案在检索结果中的排名倒数
    """
    
    def __init__(self):
        """初始化 Custom Evaluator"""
        self._backend = "custom"
    
    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str]
    ) -> Dict[str, float]:
        """
        评估检索结果的质量
        
        Args:
            query: 查询文本（此实现中不使用，但保留接口一致性）
            retrieved_ids: 检索返回的文档 ID 列表（按相关性排序）
            golden_ids: 标准答案的文档 ID 列表（ground truth）
        
        Returns:
            Dict[str, float]: 评估指标字典
                             - "hit_rate": 命中率（0.0-1.0）
                             - "mrr": 平均倒数排名（0.0-1.0）
        
        Raises:
            ValueError: 当输入参数无效时
        """
        if not retrieved_ids:
            raise ValueError("检索结果列表不能为空")
        
        if not golden_ids:
            raise ValueError("标准答案列表不能为空")
        
        # 计算 Hit Rate（命中率）
        hit_rate = self._calculate_hit_rate(retrieved_ids, golden_ids)
        
        # 计算 MRR（平均倒数排名）
        mrr = self._calculate_mrr(retrieved_ids, golden_ids)
        
        return {
            "hit_rate": hit_rate,
            "mrr": mrr
        }
    
    def _calculate_hit_rate(
        self,
        retrieved_ids: List[str],
        golden_ids: List[str]
    ) -> float:
        """
        计算命中率（Hit Rate）
        
        命中率 = 检索结果中包含的标准答案数量 / 标准答案总数
        
        Args:
            retrieved_ids: 检索返回的文档 ID 列表
            golden_ids: 标准答案的文档 ID 列表
        
        Returns:
            float: 命中率（0.0-1.0）
        """
        if not golden_ids:
            return 0.0
        
        # 将检索结果转换为集合以便快速查找
        retrieved_set = set(retrieved_ids)
        
        # 计算有多少个标准答案出现在检索结果中
        hits = sum(1 for golden_id in golden_ids if golden_id in retrieved_set)
        
        # 命中率 = 命中的标准答案数 / 标准答案总数
        return hits / len(golden_ids)
    
    def _calculate_mrr(
        self,
        retrieved_ids: List[str],
        golden_ids: List[str]
    ) -> float:
        """
        计算平均倒数排名（Mean Reciprocal Rank, MRR）
        
        MRR = 1 / rank，其中 rank 是第一个标准答案在检索结果中的位置（从1开始）
        如果没有标准答案出现在检索结果中，MRR = 0
        
        Args:
            retrieved_ids: 检索返回的文档 ID 列表（按相关性排序）
            golden_ids: 标准答案的文档 ID 列表
        
        Returns:
            float: MRR 分数（0.0-1.0）
        """
        if not golden_ids:
            return 0.0
        
        # 将标准答案转换为集合以便快速查找
        golden_set = set(golden_ids)
        
        # 找到第一个出现在检索结果中的标准答案的位置
        for rank, retrieved_id in enumerate(retrieved_ids, start=1):
            if retrieved_id in golden_set:
                # MRR = 1 / rank
                return 1.0 / rank
        
        # 如果没有标准答案出现在检索结果中，MRR = 0
        return 0.0
    
    def get_backend(self) -> str:
        """获取 backend 名称"""
        return self._backend
