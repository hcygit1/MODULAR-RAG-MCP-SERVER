"""
块边界语义修复器

扫描切分后的 chunk 列表，检测边界断裂（开头续接、末尾无终止等），
用规则拼接或 LLM 重写修复，保证每个 chunk 可独立理解。

与 BaseTransform 不同，本组件需要看到相邻块，因此以批处理方式运行。
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ingestion.models import Chunk
from src.core.settings import IngestionConfig

logger = logging.getLogger(__name__)

try:
    from src.libs.llm.base_llm import BaseLLM
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    BaseLLM = None

_TERMINAL_PUNCTUATION = frozenset("。！？.!?：:）)】」』》")
_CONTINUATION_CHARS = frozenset(",，、；;")
# 不太可能作为独立语义开头的中文虚词/助词/连词
_CN_CONTINUATION_CHARS = frozenset("的了在和与或但而且是被把将给向从对着过得地也还就都")


class BoundaryRepairer:
    """
    块边界语义修复器。

    两种模式：
    - rule：用前后块文本做规则拼接，零成本。
    - llm：把当前块 + 前后边界传给 LLM 重写，失败时降级到 rule。
    """

    def __init__(
        self,
        config: IngestionConfig,
        llm: Optional[Any] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        self._enabled = config.enable_boundary_repair
        self._mode = config.boundary_repair_mode  # "rule" | "llm"
        self._llm = llm
        self._prompt_template = prompt_template

    def repair(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[Chunk]:
        """
        批量扫描并修复块边界。

        Args:
            chunks: Transform 后的 chunk 列表（按文档顺序）
            trace: 追踪上下文（可选）

        Returns:
            修复后的 chunk 列表（长度与输入相同）
        """
        if not self._enabled or len(chunks) < 2:
            return chunks

        repaired = list(chunks)
        repair_count = 0

        for i in range(len(repaired)):
            prev_chunk = repaired[i - 1] if i > 0 else None
            next_chunk = repaired[i + 1] if i < len(repaired) - 1 else None

            if not self._needs_repair(repaired[i], prev_chunk, next_chunk):
                continue

            if self._mode == "llm" and self._llm is not None:
                repaired[i] = self._llm_repair(repaired[i], prev_chunk, next_chunk)
            else:
                repaired[i] = self._rule_repair(repaired[i], prev_chunk, next_chunk)

            repair_count += 1

        if repair_count:
            logger.info("BoundaryRepairer 修复了 %d/%d 个块", repair_count, len(chunks))

        return repaired

    # ------------------------------------------------------------------
    # Overlap 检测
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_overlap(text_a: str, text_b: str, max_check: int = 100) -> int:
        """检测 text_a 末尾与 text_b 开头的重叠字符数。"""
        limit = min(max_check, len(text_a), len(text_b))
        for length in range(limit, 0, -1):
            if text_a[-length:] == text_b[:length]:
                return length
        return 0

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_repair(
        chunk: Chunk,
        prev_chunk: Optional[Chunk] = None,
        next_chunk: Optional[Chunk] = None,
    ) -> bool:
        """启发式判断 chunk 是否存在边界断裂（跳过 overlap 区域）。"""
        text = chunk.text.strip()
        if not text:
            return False

        # --- head 检测：跳过与前块的 overlap ---
        head_broken = False
        check_head = text
        if prev_chunk:
            prev_text = prev_chunk.text.strip()
            overlap = BoundaryRepairer._detect_overlap(prev_text, text)
            check_head = text[overlap:].strip() if overlap else text

        if check_head:
            first_char = check_head[0]
            head_broken = (
                first_char.islower()
                or first_char in _CONTINUATION_CHARS
                or first_char in _CN_CONTINUATION_CHARS
            )

        # --- tail 检测：跳过与后块的 overlap ---
        tail_broken = False
        check_tail = text
        if next_chunk:
            next_text = next_chunk.text.strip()
            overlap = BoundaryRepairer._detect_overlap(text, next_text)
            check_tail = text[: len(text) - overlap].strip() if overlap else text

        if check_tail:
            last_char = check_tail[-1]
            if last_char not in _TERMINAL_PUNCTUATION:
                last_line = check_tail.split("\n")[-1].strip()
                if not last_line.startswith("#"):
                    tail_broken = True

        return head_broken or tail_broken

    # ------------------------------------------------------------------
    # 规则修复
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_repair(
        chunk: Chunk,
        prev_chunk: Optional[Chunk],
        next_chunk: Optional[Chunk],
    ) -> Chunk:
        """用前后块的非重叠边界文本做规则拼接修复。"""
        text = chunk.text.strip()
        prefix = ""
        suffix = ""

        # --- head 修复：跳过 overlap 后判断，从前块非重叠区取 prefix ---
        if prev_chunk and text:
            prev_text = prev_chunk.text.strip()
            head_overlap = BoundaryRepairer._detect_overlap(prev_text, text)
            real_start = text[head_overlap:].strip() if head_overlap else text

            if real_start:
                first_char = real_start[0]
                if (first_char.islower()
                        or first_char in _CONTINUATION_CHARS
                        or first_char in _CN_CONTINUATION_CHARS):
                    search_region = prev_text[: len(prev_text) - head_overlap].strip() if head_overlap else prev_text
                    for sep in ("。", ".", "！", "!", "？", "?"):
                        parts = search_region.rsplit(sep, 1)
                        if len(parts) > 1:
                            prefix = parts[-1].strip()
                            if prefix:
                                prefix = prefix + sep
                            break

        # --- tail 修复：跳过 overlap 后判断，从后块非重叠区取 suffix ---
        if next_chunk and text:
            next_text = next_chunk.text.strip()
            tail_overlap = BoundaryRepairer._detect_overlap(text, next_text)
            check_tail = text[: len(text) - tail_overlap].strip() if tail_overlap else text

            if check_tail and check_tail[-1] not in _TERMINAL_PUNCTUATION:
                last_line = check_tail.split("\n")[-1].strip()
                if not last_line.startswith("#"):
                    non_overlap_region = next_text[tail_overlap:].strip() if tail_overlap else next_text
                    for sep in ("。", ".", "！", "!", "？", "?"):
                        idx = non_overlap_region.find(sep)
                        if idx >= 0:
                            suffix = non_overlap_region[: idx + 1].strip()
                            break

        repaired_text = (prefix + " " + text if prefix else text)
        if suffix:
            repaired_text = repaired_text + " " + suffix
        repaired_text = repaired_text.strip()

        new_meta = chunk.metadata.copy()
        new_meta["boundary_repair"] = "rule"

        return Chunk(
            id=chunk.id,
            text=repaired_text,
            metadata=new_meta,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
        )

    # ------------------------------------------------------------------
    # LLM 修复
    # ------------------------------------------------------------------

    def _llm_repair(
        self,
        chunk: Chunk,
        prev_chunk: Optional[Chunk],
        next_chunk: Optional[Chunk],
    ) -> Chunk:
        """用 LLM 重写当前块，失败降级到规则修复。"""
        try:
            template = self._get_prompt_template()
            prev_tail = prev_chunk.text.strip()[-200:] if prev_chunk else ""
            next_head = next_chunk.text.strip()[:200] if next_chunk else ""

            prompt = template.format(
                prev_tail=prev_tail,
                current_text=chunk.text,
                next_head=next_head,
            )

            response = self._llm.chat([{"role": "user", "content": prompt}])

            if not response or not response.strip():
                logger.warning("LLM 返回空结果，降级到规则修复")
                return self._rule_repair(chunk, prev_chunk, next_chunk)

            new_meta = chunk.metadata.copy()
            new_meta["boundary_repair"] = "llm"

            return Chunk(
                id=chunk.id,
                text=response.strip(),
                metadata=new_meta,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
            )
        except Exception as e:
            logger.warning("LLM 边界修复失败，降级到规则: %s", e)
            return self._rule_repair(chunk, prev_chunk, next_chunk)

    def _get_prompt_template(self) -> str:
        if self._prompt_template:
            return self._prompt_template

        path = Path("config/prompts/boundary_repair.txt")
        if path.exists():
            self._prompt_template = path.read_text(encoding="utf-8")
            return self._prompt_template

        self._prompt_template = (
            "以下是一个文档被切分后的三个相邻片段。"
            "中间片段可能存在边界断裂（开头或结尾语义不完整）。\n\n"
            "【前一块末尾】\n{prev_tail}\n\n"
            "【当前块（需要修复）】\n{current_text}\n\n"
            "【后一块开头】\n{next_head}\n\n"
            "请修复当前块的文本，使其语义完整、可独立理解。\n"
            "要求：只输出修复后的当前块文本，不要输出其他内容。"
            "不要改变原意，只补全边界缺失的上下文。"
        )
        return self._prompt_template
