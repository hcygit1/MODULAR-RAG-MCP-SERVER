"""
ParentAggregator

将子块检索结果按 parent_id 聚合，返回完整父文档内容。
用于父子索引场景：检索打在子块上，返回整道菜（父）的完整正文。

流程：
  1. 子块结果按 parent_id 分组，计算父分数（sum / count）
  2. 按父分数降序取 top_k 个父
  3. 对每个父调 get_chunks_by_parent_id 拉全部子块，按顺序拼正文
  4. 返回父级 QueryResult 列表
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.libs.vector_store.base_vector_store import BaseVectorStore, QueryResult


class ParentAggregator:
    """
    父子索引聚合器。

    Args:
        vector_store: 向量存储实例，需支持 get_chunks_by_parent_id。
        parent_id_key: metadata 中存放 parent_id 的字段名，默认 "parent_id"。
        score_strategy: 父分数计算策略，"sum"（子分数求和）或 "count"（命中子块数）。
        chunk_separator: 合并子块正文时的分隔符。
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        parent_id_key: str = "parent_id",
        score_strategy: str = "sum",
        chunk_separator: str = "\n\n",
    ) -> None:
        self._store = vector_store
        self._parent_id_key = parent_id_key
        self._score_strategy = score_strategy
        self._separator = chunk_separator

    def aggregate(
        self,
        child_results: List[QueryResult],
        top_k: int,
        collection_name: str,
        trace: Optional[Any] = None,
    ) -> List[QueryResult]:
        """
        将子块检索结果聚合为父级结果。

        Args:
            child_results: rerank 后的子块 QueryResult 列表（已去重）
            top_k: 最终返回的父级数量
            collection_name: 集合名称，传给 get_chunks_by_parent_id
            trace: 追踪上下文（可选）

        Returns:
            List[QueryResult]: 父级结果，text 为全部子块合并正文，按父分数降序
        """
        if not child_results:
            return []

        # 步骤 1：按 parent_id 分组，id 去重，累计分数
        parent_scores: Dict[str, float] = defaultdict(float)
        parent_hit_count: Dict[str, int] = defaultdict(int)
        seen_child_ids: set = set()

        for r in child_results:
            if r.id in seen_child_ids:
                continue
            seen_child_ids.add(r.id)

            pid = r.metadata.get(self._parent_id_key)
            if not pid:
                # 没有 parent_id 的子块跳过聚合（不参与父级返回）
                continue
            parent_scores[pid] += r.score
            parent_hit_count[pid] += 1

        if not parent_scores:
            return []

        # 步骤 2：选分数函数，按父分数降序取 top_k
        if self._score_strategy == "count":
            sorted_parents = sorted(
                parent_hit_count.items(), key=lambda x: x[1], reverse=True
            )
            scored = [(pid, float(cnt)) for pid, cnt in sorted_parents[:top_k]]
        else:  # sum（默认）
            sorted_parents = sorted(
                parent_scores.items(), key=lambda x: x[1], reverse=True
            )
            scored = sorted_parents[:top_k]

        # 步骤 3：对每个父拉全部子块，拼合正文
        results: List[QueryResult] = []
        for parent_id, parent_score in scored:
            child_chunks = self._store.get_chunks_by_parent_id(
                collection_name=collection_name,
                parent_id=parent_id,
            )
            if not child_chunks:
                continue

            merged_text = self._separator.join(c.text for c in child_chunks if c.text)

            # 继承第一个子块的 metadata 作为基础
            base_meta: Dict[str, Any] = dict(child_chunks[0].metadata)

            # 合并所有子块的 image_refs（去重，保持顺序）
            all_image_refs: List[str] = []
            seen_refs: set = set()
            for c in child_chunks:
                for ref in c.metadata.get("image_refs") or []:
                    if ref not in seen_refs:
                        all_image_refs.append(ref)
                        seen_refs.add(ref)

            # 合并所有子块的 image_metadata
            all_image_metadata: List[Dict[str, Any]] = []
            seen_meta_ids: set = set()
            for c in child_chunks:
                for img_meta in c.metadata.get("image_metadata") or []:
                    img_id = img_meta.get("image_id")
                    if img_id and img_id not in seen_meta_ids:
                        all_image_metadata.append(img_meta)
                        seen_meta_ids.add(img_id)

            if all_image_refs:
                base_meta["image_refs"] = all_image_refs
            if all_image_metadata:
                base_meta["image_metadata"] = all_image_metadata

            base_meta.update({
                "parent_id": parent_id,
                "aggregated": True,
                "child_count": len(child_chunks),
                "child_chunk_ids": [c.id for c in child_chunks],
            })

            results.append(
                QueryResult(
                    id=parent_id,
                    score=round(parent_score, 6),
                    text=merged_text,
                    metadata=base_meta,
                )
            )

        return results
