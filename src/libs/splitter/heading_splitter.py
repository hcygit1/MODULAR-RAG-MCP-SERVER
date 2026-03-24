"""
HeadingSplitter 实现

按 Markdown 标题层级切分文本，输出带 parent_id 的子块。
适用于「父子索引」场景，自动检测文档结构：

模式 A（多菜文件，有 heading_split_level 标题，如 ###）：
  ## 宫保鸡丁（父）→ ### 食材 / ### 步骤（子块）
  parent_id = {doc_id}_section_{n}，每道菜各一个

模式 B（单菜文件，无 heading_split_level 标题，仅有 ##）：
  整个文档 = 一个父（parent_id = {doc_id}_section_0）
  ## 必备原料 / ## 操作 等各节 = 子块
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.settings import IngestionConfig
from src.libs.splitter.base_splitter import BaseSplitter


def _heading_level(line: str) -> Optional[int]:
    """返回该行的标题级别（1-6），非标题返回 None。"""
    m = re.match(r'^(#{1,6})\s', line)
    return len(m.group(1)) if m else None


def _heading_text(line: str) -> str:
    """提取标题正文（去掉前缀 # 符号）。"""
    return re.sub(r'^#{1,6}\s+', '', line).strip()


def _has_level(text: str, level: int) -> bool:
    """检查文档中是否存在指定级别的标题。"""
    prefix = "#" * level + " "
    for line in text.splitlines():
        if line.startswith(prefix):
            return True
    return False


class HeadingSplitter(BaseSplitter):
    """
    按 Markdown 标题层级切分，支持父子索引。
    自动检测文档结构，兼容单菜文件（模式 B）和多菜文件（模式 A）。

    每个子块的 metadata 包含：
    - parent_id: 所属父节点 id（格式 {doc_id}_section_{n}）
    - heading_text: 子块对应的标题文本
    - heading_level: 子块标题级别（int）
    """

    def __init__(self, config: IngestionConfig):
        self._config = config
        self._parent_level = config.heading_parent_level
        self._split_level = config.heading_split_level

    # ------------------------------------------------------------------
    # 核心：split_with_metadata
    # ------------------------------------------------------------------

    def split_with_metadata(
        self,
        text: str,
        doc_id: str,
        trace: Optional[Any] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        按标题层级切分，返回 (chunk_text, metadata) 列表。

        自动检测：
        - 文档含 heading_split_level（如 ###）→ 模式 A（多菜，## 是父，### 是子）
        - 文档不含 heading_split_level        → 模式 B（单菜，整文档是父，## 是子）

        Args:
            text: Markdown 全文
            doc_id: 文档 ID，用于生成 parent_id
            trace: 追踪上下文（可选）

        Returns:
            List[Tuple[str, Dict]]: 每项为 (chunk_text, metadata)
        """
        if not text:
            raise ValueError("输入文本不能为空")

        if _has_level(text, self._split_level):
            return self._split_mode_a(text, doc_id)
        else:
            return self._split_mode_b(text, doc_id)

    def _split_mode_a(
        self, text: str, doc_id: str
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        模式 A：文档含 heading_split_level 标题。
        heading_parent_level（##）是父，heading_split_level（###）是子块。
        """
        results: List[Tuple[str, Dict[str, Any]]] = []
        section_idx = 0
        current_parent_id = f"{doc_id}_section_{section_idx}"
        current_chunk_lines: List[str] = []
        current_heading_text = ""
        current_heading_level = self._split_level

        def flush_chunk() -> None:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                results.append((chunk_text, {
                    "parent_id": current_parent_id,
                    "heading_text": current_heading_text,
                    "heading_level": current_heading_level,
                }))

        for line in text.splitlines():
            level = _heading_level(line)

            if level == self._parent_level:
                flush_chunk()
                current_chunk_lines = []
                section_idx += 1
                current_parent_id = f"{doc_id}_section_{section_idx}"
                current_chunk_lines = [line]
                current_heading_text = _heading_text(line)
                current_heading_level = level

            elif level == self._split_level:
                flush_chunk()
                current_chunk_lines = [line]
                current_heading_text = _heading_text(line)
                current_heading_level = level

            else:
                current_chunk_lines.append(line)

        flush_chunk()
        return results

    def _split_mode_b(
        self, text: str, doc_id: str
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        模式 B：文档不含 heading_split_level 标题（单菜文件）。
        整个文档视为一个父（parent_id = {doc_id}_section_0），
        heading_parent_level（##）的每一节作为子块。
        """
        results: List[Tuple[str, Dict[str, Any]]] = []
        parent_id = f"{doc_id}_section_0"
        current_chunk_lines: List[str] = []
        current_heading_text = ""
        current_heading_level = self._parent_level

        def flush_chunk() -> None:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                results.append((chunk_text, {
                    "parent_id": parent_id,
                    "heading_text": current_heading_text,
                    "heading_level": current_heading_level,
                }))

        for line in text.splitlines():
            level = _heading_level(line)

            if level == self._parent_level:
                # ## 处切块
                flush_chunk()
                current_chunk_lines = [line]
                current_heading_text = _heading_text(line)
                current_heading_level = level
            else:
                current_chunk_lines.append(line)

        flush_chunk()
        return results

    # ------------------------------------------------------------------
    # split_text：兼容 BaseSplitter 接口，忽略 parent_id 信息
    # ------------------------------------------------------------------

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
    ) -> List[str]:
        """仅返回切分后的文本列表（不带 metadata）。"""
        return [chunk for chunk, _ in self.split_with_metadata(text, doc_id="__unknown__", trace=trace)]

    # ------------------------------------------------------------------
    # BaseSplitter 必须实现的抽象方法
    # ------------------------------------------------------------------

    def get_strategy(self) -> str:
        return "heading"

    def get_chunk_size(self) -> int:
        return 0  # heading 模式按结构切分，无固定大小

    def get_chunk_overlap(self) -> int:
        return 0  # heading 模式无重叠
