"""
LLM Reranker 实现

使用 LLM 对候选结果进行重排序。
读取 config/prompts/rerank.txt 构造 prompt，调用 LLM 获取排序结果。
"""
import json
import os
from pathlib import Path
from typing import List, Optional, Any, Dict

from src.libs.vector_store.base_vector_store import QueryResult
from src.libs.reranker.base_reranker import BaseReranker
from src.core.settings import Settings, RerankConfig
from src.libs.llm.llm_factory import LLMFactory


class LLMReranker(BaseReranker):
    """
    LLM Reranker 实现
    
    使用 LLM 对候选结果进行重排序。
    读取 prompt 模板，构造 prompt，调用 LLM 获取排序结果（JSON 数组格式），
    然后根据排序结果重新排列 QueryResult。
    """
    
    def __init__(self, settings: Settings, prompt_path: Optional[str] = None):
        """
        初始化 LLM Reranker
        
        Args:
            settings: 配置对象，包含 LLM 和 Rerank 配置信息
            prompt_path: Prompt 模板文件路径（可选，用于测试时注入替代文本）
                        如果为 None，则使用默认路径 config/prompts/rerank.txt
        
        Raises:
            ValueError: 当配置不完整或 prompt 文件不存在时
            RuntimeError: 当 LLM 创建失败时
        """
        self._config = settings.rerank
        self._backend = "llm"
        
        # 加载 prompt 模板
        if prompt_path is None:
            # 默认路径：相对于项目根目录
            project_root = Path(__file__).parent.parent.parent.parent
            prompt_path = project_root / "config" / "prompts" / "rerank.txt"
        else:
            prompt_path = Path(prompt_path)
        
        if not prompt_path.exists():
            raise ValueError(f"Prompt 模板文件不存在: {prompt_path}")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            self._prompt_template = f.read()
        
        # 创建 LLM 实例（使用 settings.llm 配置）
        try:
            self._llm = LLMFactory.create(settings)
        except Exception as e:
            raise RuntimeError(
                f"无法创建 LLM 实例用于 Reranker: {str(e)}"
            ) from e
    
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
            List[QueryResult]: 重排序后的结果列表，按新的相关性降序排列
        
        Raises:
            ValueError: 当查询为空或候选列表为空时
            RuntimeError: 当 LLM 调用失败或返回格式不正确时
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")
        
        if not candidates:
            raise ValueError("候选列表不能为空")
        
        # 如果只有一个候选，直接返回
        if len(candidates) == 1:
            return candidates.copy()
        
        try:
            # 构造 prompt
            prompt = self._build_prompt(query, candidates)
            
            # 调用 LLM 获取排序结果
            llm_response = self._llm.chat([
                {"role": "user", "content": prompt}
            ])
            
            # 解析 LLM 返回的 JSON 数组
            ranked_ids = self._parse_llm_response(llm_response, candidates)
            
            # 根据 ranked_ids 重新排序 candidates
            reranked_results = self._reorder_candidates(candidates, ranked_ids)
            
            return reranked_results
            
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"LLM Reranker 解析失败: LLM 返回的不是有效的 JSON 格式。"
                f"响应内容: {llm_response[:200] if 'llm_response' in locals() else 'N/A'}"
            ) from e
        except ValueError as e:
            # ValueError 可能是 _parse_llm_response 或 _reorder_candidates 抛出的
            # 检查是否是 JSON 解析相关的错误
            error_msg = str(e)
            if "JSON" in error_msg or "json" in error_msg.lower() or "解析" in error_msg or "格式不正确" in error_msg or "不是数组格式" in error_msg:
                raise RuntimeError(
                    f"LLM Reranker 解析失败: {error_msg}"
                ) from e
            else:
                raise RuntimeError(
                    f"LLM Reranker 处理失败: {error_msg}"
                ) from e
        except Exception as e:
            # 其他异常（如 LLM 调用失败）直接抛出
            raise RuntimeError(
                f"LLM Reranker 调用失败: {str(e)}"
            ) from e
    
    def _build_prompt(self, query: str, candidates: List[QueryResult]) -> str:
        """
        构造 prompt
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
        
        Returns:
            str: 格式化后的 prompt
        """
        # 格式化候选列表
        candidates_text = []
        for i, candidate in enumerate(candidates):
            candidates_text.append(
                f"[{i}] ID: {candidate.id}\n"
                f"Text: {candidate.text[:500]}"  # 限制文本长度，避免 prompt 过长
            )
        
        candidates_str = "\n\n".join(candidates_text)
        
        # 替换 prompt 模板中的占位符
        prompt = self._prompt_template.replace("{query}", query)
        prompt = prompt.replace("{candidates}", candidates_str)
        
        return prompt
    
    def _parse_llm_response(self, response: str, candidates: List[QueryResult]) -> List[str]:
        """
        解析 LLM 返回的 JSON 数组
        
        Args:
            response: LLM 返回的文本
            candidates: 候选结果列表（用于验证 ID 的有效性）
        
        Returns:
            List[str]: 排序后的 ID 列表
        
        Raises:
            ValueError: 当返回格式不正确或包含无效 ID 时
        """
        # 尝试提取 JSON 数组（可能包含在文本中）
        response = response.strip()
        
        # 首先尝试解析整个响应为 JSON
        try:
            parsed_response = json.loads(response)
            # 如果整个响应是数组，直接使用
            if isinstance(parsed_response, list):
                ranked_ids = parsed_response
            else:
                # 如果整个响应不是数组，抛出错误
                raise ValueError(
                    f"LLM 返回的不是数组格式: 期望 list，实际 {type(parsed_response).__name__}。"
                    f"响应内容: {response[:200]}"
                )
        except json.JSONDecodeError:
            # 如果整个响应不是有效的 JSON，尝试提取数组部分
            start_idx = response.find("[")
            end_idx = response.rfind("]")
            
            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                raise ValueError(
                    f"LLM 返回格式不正确: 未找到有效的 JSON 数组。"
                    f"响应内容: {response[:200]}"
                )
            
            json_str = response[start_idx:end_idx + 1]
            
            try:
                ranked_ids = json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"LLM 返回的 JSON 格式无效: {str(e)}。"
                    f"JSON 字符串: {json_str[:200]}"
                ) from e
            
            # 验证 ranked_ids 是列表
            if not isinstance(ranked_ids, list):
                raise ValueError(
                    f"LLM 返回的不是数组格式: 期望 list，实际 {type(ranked_ids).__name__}"
                )
        
        # 验证 ranked_ids 中的元素都是字符串
        if not all(isinstance(id_str, str) for id_str in ranked_ids):
            raise ValueError(
                f"LLM 返回的数组包含非字符串元素: {ranked_ids}"
            )
        
        # 验证所有 ID 都在 candidates 中
        candidate_ids = {c.id for c in candidates}
        invalid_ids = [id_str for id_str in ranked_ids if id_str not in candidate_ids]
        if invalid_ids:
            raise ValueError(
                f"LLM 返回的 ID 不在候选列表中: {invalid_ids}。"
                f"有效的候选 ID: {list(candidate_ids)}"
            )
        
        # 验证 ranked_ids 包含所有候选 ID（或者至少包含部分）
        # 如果 ranked_ids 不完整，我们仍然可以使用它，但会警告
        missing_ids = candidate_ids - set(ranked_ids)
        if missing_ids:
            # 允许部分排序，但记录警告（这里我们只返回 ranked_ids 中存在的）
            pass
        
        return ranked_ids
    
    def _reorder_candidates(
        self,
        candidates: List[QueryResult],
        ranked_ids: List[str]
    ) -> List[QueryResult]:
        """
        根据 ranked_ids 重新排序 candidates
        
        Args:
            candidates: 原始候选结果列表
            ranked_ids: LLM 返回的排序后的 ID 列表
        
        Returns:
            List[QueryResult]: 重新排序后的结果列表
        """
        # 创建 ID 到 QueryResult 的映射
        id_to_candidate = {c.id: c for c in candidates}
        
        # 按照 ranked_ids 的顺序重新排列
        reranked = []
        seen_ids = set()
        
        # 首先添加 ranked_ids 中存在的候选
        for id_str in ranked_ids:
            if id_str in id_to_candidate and id_str not in seen_ids:
                reranked.append(id_to_candidate[id_str])
                seen_ids.add(id_str)
        
        # 然后添加 ranked_ids 中不存在的候选（保持原始顺序）
        for candidate in candidates:
            if candidate.id not in seen_ids:
                reranked.append(candidate)
                seen_ids.add(candidate.id)
        
        return reranked
    
    def get_backend(self) -> str:
        """获取 backend 名称"""
        return self._backend
