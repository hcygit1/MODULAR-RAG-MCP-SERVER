"""
Metadata Enricher 实现

对 Chunk 的元数据进行语义增强：
1. 规则增强：从文本抽取/推断 title、生成简短 summary、打 tags
2. 可选 LLM 增强：使用 LLM 进一步优化元数据
3. 失败降级：LLM 不可用时回退到规则结果
"""
import re
from typing import Optional, Any, Dict, List

from src.ingestion.models import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.core.settings import IngestionConfig

# 尝试导入 BaseLLM
try:
    from src.libs.llm.base_llm import BaseLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    BaseLLM = None


class MetadataEnricher(BaseTransform):
    """
    Metadata Enricher 实现
    
    为 Chunk 生成语义元数据（title、summary、tags），提升检索质量。
    支持规则增强和可选的 LLM 增强。
    """
    
    def __init__(
        self,
        config: IngestionConfig,
        llm: Optional[Any] = None
    ):
        """
        初始化 MetadataEnricher
        
        Args:
            config: Ingestion 配置对象，包含 enable_metadata_enrichment 等开关
            llm: LLM 实例（可选），用于 LLM 增强功能
        """
        self._config = config
        self._enable_llm = config.enable_metadata_enrichment
        self._llm = llm
    
    def transform(
        self,
        chunk: Chunk,
        trace: Optional[Any] = None
    ) -> Chunk:
        """
        对 Chunk 的元数据进行增强
        
        Args:
            chunk: 输入的 Chunk 对象
            trace: 追踪上下文（可选）
        
        Returns:
            Chunk: 处理后的 Chunk 对象，metadata 中包含 title、summary、tags
        
        Raises:
            ValueError: 当 Chunk 无效时
        """
        if not chunk or not chunk.text:
            raise ValueError("Chunk 或 Chunk.text 不能为空")
        
        # 步骤 1: 规则增强（始终执行）
        enriched_metadata = self._rule_based_enrichment(chunk.text, chunk.metadata)
        
        # 步骤 2: 可选 LLM 增强
        if self._enable_llm and self._llm is not None:
            try:
                llm_enriched = self._llm_enrichment(chunk.text, enriched_metadata)
                # 合并 LLM 增强的结果（LLM 结果优先）
                enriched_metadata.update(llm_enriched)
                enriched_metadata["enrichment_method"] = "llm"
            except Exception as e:
                # LLM 调用失败，降级到规则结果
                enriched_metadata["enrichment_method"] = "rule_fallback"
                enriched_metadata["enrichment_fallback_reason"] = str(e)
                # 继续使用规则增强的结果
        else:
            enriched_metadata["enrichment_method"] = "rule"
        
        # 确保必需的字段存在
        if "title" not in enriched_metadata or not enriched_metadata["title"]:
            enriched_metadata["title"] = self._extract_title_fallback(chunk.text)
        if "summary" not in enriched_metadata or not enriched_metadata["summary"]:
            enriched_metadata["summary"] = self._generate_summary_fallback(chunk.text)
        if "tags" not in enriched_metadata:
            enriched_metadata["tags"] = []
        
        # 创建新的 Chunk（保留原始文本和定位信息）
        enriched_chunk = Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata={**chunk.metadata, **enriched_metadata},  # 合并元数据
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset
        )
        
        return enriched_chunk
    
    def _rule_based_enrichment(
        self,
        text: str,
        existing_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        基于规则的元数据增强
        
        从文本中提取 title、生成 summary、推断 tags。
        
        Args:
            text: Chunk 文本内容
            existing_metadata: 现有的元数据
        
        Returns:
            Dict[str, Any]: 增强后的元数据
        """
        enriched = {}
        
        # 1. 提取 Title
        title = self._extract_title(text, existing_metadata)
        enriched["title"] = title
        
        # 2. 生成 Summary
        summary = self._generate_summary(text)
        enriched["summary"] = summary
        
        # 3. 推断 Tags
        tags = self._infer_tags(text)
        enriched["tags"] = tags
        
        return enriched
    
    def _extract_title(
        self,
        text: str,
        existing_metadata: Dict[str, Any]
    ) -> str:
        """
        提取标题
        
        优先级：
        1. 从 Markdown 标题（# 开头）提取
        2. 从现有 metadata 中获取
        3. 从文本第一行提取
        4. 生成默认标题
        
        Args:
            text: Chunk 文本
            existing_metadata: 现有元数据
        
        Returns:
            str: 标题
        """
        # 优先级 1: 从现有 metadata 获取
        if "title" in existing_metadata and existing_metadata["title"]:
            return str(existing_metadata["title"])
        
        # 优先级 2: 从 Markdown 标题提取
        lines = text.split('\n')
        for line in lines[:5]:  # 只检查前 5 行
            line = line.strip()
            # 匹配 Markdown 标题：## Title 或 # Title
            if line.startswith('#'):
                title = re.sub(r'^#+\s*', '', line).strip()
                if title:
                    return title[:100]  # 限制长度
        
        # 优先级 3: 从第一行提取（如果不是空行）
        first_line = lines[0].strip() if lines else ""
        if first_line and len(first_line) > 5 and len(first_line) < 100:
            # 如果第一行看起来像标题（长度适中，不以标点结尾）
            if not first_line.endswith(('.', '。', '!', '!', '?', '?')):
                return first_line[:100]
        
        # 优先级 4: 生成默认标题
        return self._extract_title_fallback(text)
    
    def _extract_title_fallback(self, text: str) -> str:
        """生成默认标题（fallback）"""
        # 取文本前 50 个字符作为标题
        preview = text[:50].strip()
        if preview:
            # 去除换行和多余空白
            preview = re.sub(r'\s+', ' ', preview)
            return preview
        return "Untitled Chunk"
    
    def _generate_summary(self, text: str) -> str:
        """
        生成摘要
        
        规则：取文本的前几个句子或前 N 个字符。
        
        Args:
            text: Chunk 文本
        
        Returns:
            str: 摘要
        """
        # 清理文本
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # 尝试提取前 2-3 个句子
        sentences = re.split(r'[.!?。！？]\s+', cleaned)
        if len(sentences) >= 2:
            summary = '. '.join(sentences[:2])
            if len(summary) <= 200:
                return summary + '.'
        
        # 如果句子提取失败，取前 150 个字符
        if len(cleaned) > 150:
            summary = cleaned[:150]
            # 尝试在最后一个完整单词处截断
            last_space = summary.rfind(' ')
            if last_space > 100:
                summary = summary[:last_space]
            return summary + '...'
        
        return cleaned
    
    def _generate_summary_fallback(self, text: str) -> str:
        """生成默认摘要（fallback）"""
        cleaned = re.sub(r'\s+', ' ', text.strip())
        if len(cleaned) > 100:
            return cleaned[:100] + '...'
        return cleaned if cleaned else "No summary available"
    
    def _infer_tags(self, text: str) -> List[str]:
        """
        推断标签
        
        基于关键词和常见模式推断标签。
        
        Args:
            text: Chunk 文本
        
        Returns:
            List[str]: 标签列表
        """
        tags = []
        text_lower = text.lower()
        
        # 常见技术关键词映射
        keyword_tags = {
            "python": ["python", "编程"],
            "架构": ["architecture", "架构", "设计"],
            "api": ["api", "接口"],
            "数据库": ["database", "数据库", "sql"],
            "机器学习": ["machine learning", "ml", "机器学习", "ai"],
            "深度学习": ["deep learning", "深度学习", "neural"],
            "前端": ["frontend", "前端", "react", "vue"],
            "后端": ["backend", "后端", "server"],
            "部署": ["deployment", "部署", "docker"],
            "测试": ["test", "测试", "testing"],
        }
        
        # 检查关键词
        for tag, keywords in keyword_tags.items():
            for keyword in keywords:
                if keyword in text_lower:
                    tags.append(tag)
                    break
        
        # 如果没找到标签，尝试从文本中提取重要名词
        if not tags:
            # 简单的名词提取（大写开头的单词，可能是专有名词）
            words = re.findall(r'\b[A-Z][a-z]+\b', text)
            if words:
                # 取前 3 个作为标签
                tags = list(set(words[:3]))
        
        # 限制标签数量
        return tags[:5]
    
    def _llm_enrichment(
        self,
        text: str,
        rule_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用 LLM 进行元数据增强
        
        Args:
            text: Chunk 文本
            rule_metadata: 规则增强的结果
        
        Returns:
            Dict[str, Any]: LLM 增强的元数据
        
        Raises:
            RuntimeError: 当 LLM 调用失败时
        """
        if not LLM_AVAILABLE or self._llm is None:
            raise RuntimeError("LLM 不可用")
        
        # 构建 Prompt（限制文本长度避免 token 过多）
        text_preview = text[:1000]
        prompt = f"""请为以下文档片段生成元数据：

文档片段：
{text_preview}

请以 JSON 格式返回：
{{
  "title": "精准的小标题（不超过 50 字）",
  "summary": "内容摘要（不超过 200 字）",
  "tags": ["标签1", "标签2", "标签3"]
}}

要求：
- title: 简洁准确地概括片段主题
- summary: 概括片段的核心内容
- tags: 3-5 个相关主题标签

只返回 JSON，不要其他内容。"""
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._llm.chat(messages)
            
            # 尝试解析 JSON 响应
            import json
            # 提取 JSON 部分（可能包含 markdown 代码块）
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                llm_metadata = json.loads(json_str)
                
                # 验证必需字段
                result = {
                    "title": llm_metadata.get("title", rule_metadata.get("title", "")),
                    "summary": llm_metadata.get("summary", rule_metadata.get("summary", "")),
                    "tags": llm_metadata.get("tags", rule_metadata.get("tags", []))
                }
                
                # 验证非空
                if not result["title"]:
                    result["title"] = rule_metadata.get("title", "")
                if not result["summary"]:
                    result["summary"] = rule_metadata.get("summary", "")
                if not result["tags"]:
                    result["tags"] = rule_metadata.get("tags", [])
                
                return result
            else:
                # 如果无法解析 JSON，返回规则结果
                raise ValueError("LLM 返回格式无效")
        except Exception as e:
            raise RuntimeError(f"LLM 增强失败: {str(e)}") from e
    
    def get_transform_name(self) -> str:
        """获取 Transform 名称"""
        return "metadata_enricher"
