"""
Cross-Encoder Reranker 实现

使用 Cross-Encoder 模型对候选结果进行重排序。
支持本地模型和托管模型，提供超时和失败回退机制。
"""
import signal
import threading
from typing import List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from src.libs.vector_store.base_vector_store import QueryResult
from src.libs.reranker.base_reranker import BaseReranker
from src.core.settings import RerankConfig

# 尝试导入 CrossEncoder
try:
    from sentence_transformers import CrossEncoder  # type: ignore
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder Reranker 实现
    
    使用 Cross-Encoder 模型对候选结果进行重排序。
    Cross-Encoder 会对 query 和每个 candidate 进行联合编码，输出相关性分数。
    """
    
    def __init__(
        self,
        config: RerankConfig,
        scorer: Optional[Callable[[str, str], float]] = None
    ):
        """
        初始化 Cross-Encoder Reranker
        
        Args:
            config: Rerank 配置对象，包含 model、top_m、timeout_seconds 等
            scorer: 可选的打分函数（用于测试），格式为 (query: str, text: str) -> float
                    如果提供，将使用此函数而不是真实的 CrossEncoder 模型
        
        Raises:
            ValueError: 当配置不完整时
            RuntimeError: 当 CrossEncoder 未安装且未提供 scorer 时
        """
        if not config.model:
            raise ValueError("Cross-Encoder 模型名称不能为空")
        
        self._config = config
        self._backend = "cross_encoder"
        self._model_name = config.model
        self._top_m = config.top_m
        self._timeout_seconds = config.timeout_seconds
        
        # 如果提供了 scorer（用于测试），使用它
        if scorer is not None:
            self._scorer = scorer
            self._use_mock = True
        else:
            # 检查 CrossEncoder 是否可用
            if not CROSS_ENCODER_AVAILABLE:
                raise RuntimeError(
                    "CrossEncoder 未安装。请安装: pip install sentence-transformers"
                )
            
            # 初始化 CrossEncoder 模型
            try:
                self._model = CrossEncoder(self._model_name)
            except Exception as e:
                raise RuntimeError(
                    f"无法加载 Cross-Encoder 模型 '{self._model_name}': {str(e)}"
                ) from e
            
            self._scorer = None
            self._use_mock = False
    
    def rerank(
        self,
        query: str,
        candidates: List[QueryResult],
        trace: Optional[Any] = None
    ) -> List[QueryResult]:
        """
        对候选结果进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选结果列表，已按初始分数排序
            trace: 追踪上下文（可选）
        
        Returns:
            List[QueryResult]: 重排序后的结果列表，按新的相关性分数降序排列
        
        Raises:
            ValueError: 当查询为空或候选列表为空时
            RuntimeError: 当重排序操作失败或超时时
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")
        
        if not candidates:
            raise ValueError("候选列表不能为空")
        
        # 如果只有一个候选，直接返回
        if len(candidates) == 1:
            return candidates.copy()
        
        # 限制候选数量为 top_m
        top_candidates = candidates[:self._top_m] if len(candidates) > self._top_m else candidates
        
        try:
            # 使用超时机制进行打分
            scores = self._score_with_timeout(query, top_candidates)
            
            # 创建 (score, candidate) 对并排序
            scored_candidates = [
                (score, candidate) for score, candidate in zip(scores, top_candidates)
            ]
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # 构建重排序后的结果列表，更新分数
            reranked_results = [
                QueryResult(
                    id=candidate.id,
                    score=score,
                    text=candidate.text,
                    metadata=candidate.metadata.copy() if candidate.metadata else {}
                )
                for score, candidate in scored_candidates
            ]
            
            return reranked_results
            
        except FutureTimeoutError:
            raise RuntimeError(
                f"Cross-Encoder Reranker 超时（{self._timeout_seconds}秒）。"
                f"请考虑增加 timeout_seconds 或减少 top_m。"
            )
        except Exception as e:
            # 其他异常（如模型错误）直接抛出
            raise RuntimeError(
                f"Cross-Encoder Reranker 调用失败: {str(e)}"
            ) from e
    
    def _score_with_timeout(
        self,
        query: str,
        candidates: List[QueryResult]
    ) -> List[float]:
        """
        使用超时机制对候选进行打分
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
        
        Returns:
            List[float]: 相关性分数列表
        
        Raises:
            FutureTimeoutError: 当操作超时时
        """
        def score_candidates():
            """内部函数：执行打分操作"""
            if self._use_mock and self._scorer:
                # 使用 mock scorer（用于测试）
                return [self._scorer(query, candidate.text) for candidate in candidates]
            else:
                # 使用真实的 CrossEncoder 模型
                # 准备输入对：[(query, candidate_text), ...]
                pairs = [(query, candidate.text) for candidate in candidates]
                scores = self._model.predict(pairs)
                # CrossEncoder.predict 返回 numpy array，转换为 list
                return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        
        # 使用 ThreadPoolExecutor 实现超时（对 mock scorer 和真实模型都适用）
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(score_candidates)
            try:
                scores = future.result(timeout=self._timeout_seconds)
                return scores
            except FutureTimeoutError:
                # 取消任务
                future.cancel()
                raise
    
    def get_backend(self) -> str:
        """获取 backend 名称"""
        return self._backend
