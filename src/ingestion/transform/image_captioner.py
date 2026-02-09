"""
Image Captioner 实现

对 Chunk 中的图片引用生成描述（Caption）：
1. 检查 image_refs：如果存在图片引用，尝试生成描述
2. 可选 Vision LLM 增强：使用 Vision LLM 生成图片描述
3. 失败降级：Vision LLM 不可用时保留 image_refs，标记 has_unprocessed_images
"""
import os
from pathlib import Path
from typing import Optional, Any, Dict, List

from src.ingestion.models import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.core.settings import IngestionConfig, VisionLLMConfig

# 尝试导入 BaseLLM（Vision LLM 可能也使用类似的接口）
try:
    from src.libs.llm.base_llm import BaseLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    BaseLLM = None


class ImageCaptioner(BaseTransform):
    """
    Image Captioner 实现
    
    为 Chunk 中的图片引用生成文本描述，提升多模态检索能力。
    支持可选的 Vision LLM 增强和失败降级。
    """
    
    def __init__(
        self,
        config: IngestionConfig,
        vision_llm_config: Optional[VisionLLMConfig] = None,
        vision_llm: Optional[Any] = None,
        prompt_template: Optional[str] = None
    ):
        """
        初始化 ImageCaptioner
        
        Args:
            config: Ingestion 配置对象，包含 enable_image_captioning 等开关
            vision_llm_config: Vision LLM 配置对象（可选）
            vision_llm: Vision LLM 实例（可选），用于生成图片描述
            prompt_template: Prompt 模板（可选），用于自定义图片描述生成
        """
        self._config = config
        self._enable_captioning = config.enable_image_captioning
        self._vision_llm_config = vision_llm_config
        self._vision_llm = vision_llm
        self._prompt_template = prompt_template
    
    def transform(
        self,
        chunk: Chunk,
        trace: Optional[Any] = None
    ) -> Chunk:
        """
        对 Chunk 中的图片引用生成描述
        
        Args:
            chunk: 输入的 Chunk 对象
            trace: 追踪上下文（可选）
        
        Returns:
            Chunk: 处理后的 Chunk 对象，metadata 中包含图片描述或降级标记
        
        Raises:
            ValueError: 当 Chunk 无效时
        """
        if not chunk or not chunk.text:
            raise ValueError("Chunk 或 Chunk.text 不能为空")
        
        # 检查是否有图片引用
        image_refs = chunk.metadata.get("image_refs", [])
        
        if not image_refs:
            # 没有图片引用，直接返回
            return chunk
        
        # 检查是否启用图片描述生成
        if not self._enable_captioning:
            # 配置禁用，走降级路径
            return self._apply_fallback(chunk, image_refs, reason="image_captioning_disabled")
        
        # 检查 Vision LLM 是否可用
        if self._vision_llm is None:
            # Vision LLM 不可用，走降级路径
            return self._apply_fallback(chunk, image_refs, reason="vision_llm_not_available")
        
        # 尝试生成图片描述
        try:
            captions = self._generate_captions(chunk, image_refs)
            
            # 检查是否生成了有效的描述
            valid_captions = {
                k: v for k, v in captions.items() 
                if v and not v.startswith("[图片描述生成失败")
            }
            
            if not valid_captions:
                # 所有图片描述生成都失败，走降级路径
                # 检查失败原因
                if captions:
                    # 有失败记录，使用第一个失败原因
                    first_failure = next(iter(captions.values()))
                    if "Vision LLM 调用失败" in first_failure:
                        reason = f"vision_llm_error: {first_failure}"
                    else:
                        reason = "all_captions_generation_failed"
                else:
                    reason = "all_captions_generation_failed"
                
                return self._apply_fallback(
                    chunk,
                    image_refs,
                    reason=reason
                )
            
            # 将描述写入 metadata（包含所有描述，包括失败的）
            # 这样调用者可以看到哪些图片成功，哪些失败
            enriched_metadata = chunk.metadata.copy()
            enriched_metadata["image_captions"] = captions  # 包含所有，包括失败的
            enriched_metadata["captioning_method"] = "vision_llm"
            
            # 可选：将描述注入到文本中（只注入有效的描述）
            enriched_text = self._inject_captions_to_text(chunk.text, valid_captions, image_refs)
            
            # 创建新的 Chunk
            enriched_chunk = Chunk(
                id=chunk.id,
                text=enriched_text,
                metadata=enriched_metadata,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset
            )
            
            return enriched_chunk
            
        except RuntimeError as e:
            # Vision LLM 调用失败（RuntimeError），走降级路径
            return self._apply_fallback(
                chunk, 
                image_refs, 
                reason=f"vision_llm_error: {str(e)}"
            )
        except Exception as e:
            # 其他异常，走降级路径
            return self._apply_fallback(
                chunk, 
                image_refs, 
                reason=f"unexpected_error: {str(e)}"
            )
    
    def _generate_captions(
        self,
        chunk: Chunk,
        image_refs: List[str]
    ) -> Dict[str, str]:
        """
        为图片引用生成描述
        
        Args:
            chunk: Chunk 对象，包含上下文信息
            image_refs: 图片引用列表
        
        Returns:
            Dict[str, str]: 图片 ID 到描述的映射
        
        Raises:
            RuntimeError: 当 Vision LLM 完全不可用时
        """
        if not LLM_AVAILABLE or self._vision_llm is None:
            raise RuntimeError("Vision LLM 不可用")
        
        captions = {}
        prompt_template = self._load_prompt_template()
        
        # 获取上下文文本（图片前后的文本）
        context_text = self._extract_context_text(chunk.text, image_refs)
        
        for image_id in image_refs:
            try:
                # 构建 Prompt
                prompt = prompt_template.format(
                    context_text=context_text[:500]  # 限制上下文长度
                )
                
                # 调用 Vision LLM 生成描述
                # 注意：这里假设 Vision LLM 有一个方法可以处理图片
                # 实际实现可能需要根据 Vision LLM 的具体接口调整
                caption = self._call_vision_llm(image_id, prompt)
                
                if caption and caption.strip():
                    captions[image_id] = caption.strip()
                else:
                    # LLM 返回空结果，使用默认描述
                    captions[image_id] = f"[图片: {image_id}]"
                    
            except Exception as e:
                # 单个图片处理失败，记录错误但继续处理其他图片
                # 如果所有图片都失败，外层会检测并降级
                captions[image_id] = f"[图片描述生成失败: {str(e)}]"
                # 如果第一个图片就失败且是 RuntimeError，可能需要重新抛出
                # 但为了支持部分成功，这里只记录错误
        
        return captions
    
    def _call_vision_llm(
        self,
        image_id: str,
        prompt: str
    ) -> str:
        """
        调用 Vision LLM 生成图片描述
        
        Args:
            image_id: 图片 ID
            prompt: 提示文本
        
        Returns:
            str: 图片描述
        
        Raises:
            RuntimeError: 当 Vision LLM 调用失败时
        """
        if self._vision_llm is None:
            raise RuntimeError("Vision LLM 不可用")
        
        # 尝试获取图片路径
        # 注意：这里假设图片路径可以从 image_id 推导或从 metadata 获取
        # 实际实现可能需要根据项目的图片存储策略调整
        image_path = self._get_image_path(image_id)
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # 如果 Vision LLM 支持图片输入（如 DashScopeVisionLLM），传递图片路径
        try:
            # 检查 Vision LLM 是否支持 image_path 参数
            if hasattr(self._vision_llm, 'chat') and image_path:
                # DashScopeVisionLLM 的 chat 方法支持 image_path 参数
                if hasattr(self._vision_llm, '_add_image_to_messages'):
                    # 使用支持图片的 chat 方法
                    response = self._vision_llm.chat(messages, image_path=image_path)
                else:
                    # 普通 LLM，只传递消息
                    response = self._vision_llm.chat(messages)
            else:
                # 普通 LLM 接口
                response = self._vision_llm.chat(messages)
            return response
        except Exception as e:
            raise RuntimeError(f"Vision LLM 调用失败: {str(e)}") from e
    
    def _get_image_path(self, image_id: str) -> Optional[str]:
        """
        根据 image_id 获取图片路径
        
        Args:
            image_id: 图片 ID
        
        Returns:
            Optional[str]: 图片路径，如果找不到则返回 None
        """
        # 这里需要根据项目的图片存储策略实现
        # 可能的策略：
        # 1. 从 metadata.images 列表中查找
        # 2. 根据 image_id 推导路径（如 data/images/{image_id}.png）
        # 3. 从文档的 images 元数据中查找
        
        # 暂时返回 None，实际实现时需要根据项目结构调整
        return None
    
    def _extract_context_text(
        self,
        text: str,
        image_refs: List[str]
    ) -> str:
        """
        提取图片周围的上下文文本
        
        Args:
            text: Chunk 文本
            image_refs: 图片引用列表
        
        Returns:
            str: 上下文文本
        """
        # 简单实现：返回整个文本作为上下文
        # 更精细的实现可以提取图片占位符前后的段落
        return text
    
    def _inject_captions_to_text(
        self,
        text: str,
        captions: Dict[str, str],
        image_refs: List[str]
    ) -> str:
        """
        将图片描述注入到文本中
        
        Args:
            text: 原始文本
            captions: 图片描述字典
            image_refs: 图片引用列表
        
        Returns:
            str: 注入描述后的文本
        """
        # 将描述追加到图片占位符后面
        enriched_text = text
        for image_id in image_refs:
            if image_id in captions:
                caption = captions[image_id]
                # 查找图片占位符并追加描述
                placeholder = f"[IMAGE: {image_id}]"
                if placeholder in enriched_text:
                    enriched_text = enriched_text.replace(
                        placeholder,
                        f"{placeholder}\n[图片描述: {caption}]"
                    )
        
        return enriched_text
    
    def _apply_fallback(
        self,
        chunk: Chunk,
        image_refs: List[str],
        reason: str
    ) -> Chunk:
        """
        应用降级策略：保留 image_refs，标记 has_unprocessed_images
        
        Args:
            chunk: 原始 Chunk
            image_refs: 图片引用列表
            reason: 降级原因
        
        Returns:
            Chunk: 降级后的 Chunk
        """
        enriched_metadata = chunk.metadata.copy()
        enriched_metadata["has_unprocessed_images"] = True
        enriched_metadata["captioning_method"] = "fallback"
        enriched_metadata["captioning_fallback_reason"] = reason
        
        # 保留 image_refs
        enriched_metadata["image_refs"] = image_refs
        
        # 创建新的 Chunk（保留原始文本）
        fallback_chunk = Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=enriched_metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset
        )
        
        return fallback_chunk
    
    def _load_prompt_template(self) -> str:
        """
        加载 Prompt 模板
        
        Returns:
            str: Prompt 模板文本
        
        Raises:
            FileNotFoundError: 当模板文件不存在时
            IOError: 当模板文件读取失败时
        """
        if self._prompt_template:
            return self._prompt_template
        
        # 从文件加载（文件应该存在，因为它是项目的一部分）
        prompt_file = Path("config/prompts/image_captioning.txt")
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt 模板文件不存在: {prompt_file}. "
                "请确保 config/prompts/image_captioning.txt 文件存在。"
            )
        
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                template = f.read()
                if not template.strip():
                    raise ValueError(f"Prompt 模板文件为空: {prompt_file}")
                return template
        except Exception as e:
            raise IOError(f"读取 Prompt 模板文件失败: {prompt_file}. 错误: {str(e)}") from e
    
    def get_transform_name(self) -> str:
        """获取 Transform 名称"""
        return "image_captioner"
