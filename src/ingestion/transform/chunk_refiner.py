"""
Chunk Refiner 实现

对 Chunk 进行文本清理和智能重组：
1. 规则去噪：去除页眉页脚、页码、多余空白等噪声
2. 可选 LLM 重写：使用 LLM 进一步优化文本，确保语义完整
3. 失败降级：LLM 不可用时回退到规则结果
"""
import re
import os
from pathlib import Path
from typing import Optional, Any

from src.ingestion.models import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.core.settings import Settings, IngestionConfig

# 尝试导入 BaseLLM
try:
    from src.libs.llm.base_llm import BaseLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    BaseLLM = None


class ChunkRefiner(BaseTransform):
    """
    Chunk Refiner 实现
    
    对 Chunk 文本进行清理和优化，提升检索质量。
    支持规则去噪和可选的 LLM 重写。
    """
    
    def __init__(
        self,
        config: IngestionConfig,
        llm: Optional[Any] = None,
        prompt_template: Optional[str] = None
    ):
        """
        初始化 ChunkRefiner
        
        Args:
            config: Ingestion 配置对象，包含 enable_llm_refinement 等开关
            llm: LLM 实例（可选），用于 LLM 重写功能
            prompt_template: Prompt 模板（可选），如果不提供则从文件读取
        """
        self._config = config
        self._enable_llm = config.enable_llm_refinement
        self._llm = llm
        self._prompt_template = prompt_template or self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """
        从文件加载 Prompt 模板
        
        Returns:
            str: Prompt 模板内容
        """
        prompt_path = Path("config/prompts/chunk_refinement.txt")
        if prompt_path.exists():
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                # 如果读取失败，返回默认模板
                return self._get_default_prompt_template()
        else:
            return self._get_default_prompt_template()
    
    def _get_default_prompt_template(self) -> str:
        """获取默认 Prompt 模板"""
        return """You are a text refinement assistant. Your task is to improve the quality of document chunks for better retrieval.

Given a chunk of text from a document, refine it by:
1. Removing noise (headers, footers, page numbers, etc.)
2. Merging logically related but physically separated content
3. Ensuring each chunk is self-contained and semantically complete
4. Preserving all important information

Rules:
- Do not add information not present in the original
- Maintain the original meaning and tone
- Keep technical terms and proper nouns intact
- Ensure the output is coherent and readable

Original chunk:
{chunk_text}

Refined chunk:"""
    
    def transform(
        self,
        chunk: Chunk,
        trace: Optional[Any] = None
    ) -> Chunk:
        """
        对 Chunk 进行清理和优化
        
        Args:
            chunk: 输入的 Chunk 对象
            trace: 追踪上下文（可选）
        
        Returns:
            Chunk: 处理后的 Chunk 对象
        
        Raises:
            ValueError: 当 Chunk 无效时
        """
        if not chunk or not chunk.text:
            raise ValueError("Chunk 或 Chunk.text 不能为空")
        
        # 步骤 1: 规则去噪（始终执行）
        refined_text = self._rule_based_cleanup(chunk.text)
        
        # 步骤 2: 可选 LLM 重写
        if self._enable_llm and self._llm is not None:
            try:
                llm_refined_text = self._llm_refinement(refined_text)
                refined_text = llm_refined_text
                
                # 记录处理方式
                chunk.metadata["refinement_method"] = "llm"
            except Exception as e:
                # LLM 调用失败，降级到规则结果
                chunk.metadata["refinement_method"] = "rule_fallback"
                chunk.metadata["refinement_fallback_reason"] = str(e)
                # 继续使用规则去噪的结果
        else:
            chunk.metadata["refinement_method"] = "rule"
        
        # 创建新的 Chunk（保留原始定位信息）
        refined_chunk = Chunk(
            id=chunk.id,
            text=refined_text,
            metadata=chunk.metadata.copy(),
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset
        )
        
        return refined_chunk
    
    def _rule_based_cleanup(self, text: str) -> str:
        """
        基于规则的文本清理
        
        去除页眉页脚、页码、多余空白等噪声。
        
        Args:
            text: 原始文本
        
        Returns:
            str: 清理后的文本
        """
        if not text:
            return text
        
        # 1. 去除页眉页脚模式（如 "第 X 页"、"Page X"、"第 X 页/共 Y 页"）
        text = re.sub(r'第\s*\d+\s*页', '', text)
        text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'第\s*\d+\s*页\s*/\s*共\s*\d+\s*页', '', text)
        text = re.sub(r'\d+\s*/\s*\d+', '', text)  # 页码格式如 "1 / 10"
        
        # 2. 去除常见的页眉页脚标记
        text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)  # "- 1 -"
        text = re.sub(r'^\s*_\s*\d+\s*_\s*$', '', text, flags=re.MULTILINE)  # "_ 1 _"
        
        # 3. 去除多余的空行（保留单个空行，去除连续多个空行）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 4. 去除行首行尾的空白字符
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        text = '\n'.join(cleaned_lines)
        
        # 5. 去除连续的空格（保留单个空格）
        text = re.sub(r' {2,}', ' ', text)
        
        # 6. 去除文本开头和结尾的空白
        text = text.strip()
        
        return text
    
    def _llm_refinement(self, text: str) -> str:
        """
        使用 LLM 进行文本重写
        
        Args:
            text: 已清理的文本
        
        Returns:
            str: LLM 重写后的文本
        
        Raises:
            RuntimeError: 当 LLM 调用失败时
        """
        if not LLM_AVAILABLE or self._llm is None:
            raise RuntimeError("LLM 不可用")
        
        # 构建 Prompt
        prompt = self._prompt_template.format(chunk_text=text)
        
        # 调用 LLM
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            refined_text = self._llm.chat(messages)
            
            # 验证输出不为空
            if not refined_text or not refined_text.strip():
                raise RuntimeError("LLM 返回空结果")
            
            return refined_text.strip()
        except Exception as e:
            raise RuntimeError(f"LLM 重写失败: {str(e)}") from e
    
    def get_transform_name(self) -> str:
        """获取 Transform 名称"""
        return "chunk_refiner"
