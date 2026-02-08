"""
Evaluator 工厂模块

根据配置创建对应的 Evaluator 实现实例。
支持通过配置文件切换不同的评估框架，无需修改代码。
"""
from typing import List

from src.core.settings import Settings
from src.libs.evaluator.base_evaluator import BaseEvaluator


class EvaluatorFactory:
    """
    Evaluator 工厂类
    
    根据配置动态创建对应的 Evaluator 实现。
    支持组合模式：可以同时创建多个 Evaluator 实例。
    使用工厂模式实现配置驱动的组件选择。
    """
    
    @staticmethod
    def create(settings: Settings) -> List[BaseEvaluator]:
        """
        根据配置创建 Evaluator 实例列表（支持组合模式）
        
        Args:
            settings: 配置对象，包含 Evaluation 配置信息
        
        Returns:
            List[BaseEvaluator]: Evaluator 实例列表
        
        Raises:
            ValueError: 当 backend 不支持或配置不完整时
            NotImplementedError: 当 backend 的实现尚未完成时
        """
        config = settings.evaluation
        backends = config.backends
        
        evaluators = []
        
        for backend in backends:
            backend_lower = backend.lower()
            
            if backend_lower == "custom":
                # B6 阶段实现
                from src.libs.evaluator.custom_evaluator import CustomEvaluator
                evaluators.append(CustomEvaluator())
            elif backend_lower == "ragas":
                # 未来实现
                raise NotImplementedError(
                    "Ragas Evaluator 实现尚未完成。"
                    "请先使用其他 backend 或等待实现。"
                )
            elif backend_lower == "deepeval":
                # 未来实现
                raise NotImplementedError(
                    "DeepEval Evaluator 实现尚未完成。"
                    "请先使用其他 backend 或等待实现。"
                )
            else:
                raise ValueError(
                    f"不支持的 Evaluator backend: {backend}。"
                    f"支持的 backend: custom, ragas, deepeval"
                )
        
        return evaluators
    
    @staticmethod
    def create_custom() -> BaseEvaluator:
        """
        创建 CustomEvaluator 实例（用于测试）
        
        Returns:
            BaseEvaluator: CustomEvaluator 实例
        """
        from src.libs.evaluator.custom_evaluator import CustomEvaluator
        return CustomEvaluator()
