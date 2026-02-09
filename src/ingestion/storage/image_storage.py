"""
Image Storage 实现

负责保存图片文件到文件系统，并维护 image_id→path 映射索引。
"""
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List
from base64 import b64decode


class ImageStorage:
    """
    Image Storage 实现
    
    负责将图片保存到文件系统，并维护 image_id 到文件路径的映射索引。
    支持图片保存、路径查询和索引管理。
    """
    
    def __init__(self, base_path: str = "data/images"):
        """
        初始化 ImageStorage
        
        Args:
            base_path: 图片存储的基础路径，默认为 "data/images"
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        
        # 内存中的索引：image_id -> image_info
        self._image_index: Dict[str, Dict[str, Any]] = {}
    
    def save_image(
        self,
        image_id: str,
        image_data: bytes,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None
    ) -> str:
        """
        保存图片文件并记录映射
        
        Args:
            image_id: 图片唯一标识符
            image_data: 图片二进制数据
            collection_name: 集合名称，用于组织图片文件
            metadata: 图片元数据（可选），例如 source_doc, page, width, height, mime_type
            trace: 追踪上下文（可选）
        
        Returns:
            str: 保存后的图片文件路径（相对路径或绝对路径）
        
        Raises:
            ValueError: 当参数无效时
            RuntimeError: 当保存失败时
        """
        if not image_id:
            raise ValueError("image_id 不能为空")
        
        if not image_data:
            raise ValueError("image_data 不能为空")
        
        if not collection_name:
            raise ValueError("collection_name 不能为空")
        
        # 创建集合目录
        collection_path = self._base_path / collection_name
        collection_path.mkdir(parents=True, exist_ok=True)
        
        # 确定文件扩展名（从 metadata 或默认使用 .png）
        mime_type = (metadata or {}).get("mime_type", "image/png")
        extension = self._get_extension_from_mime_type(mime_type)
        
        # 构建文件路径
        filename = f"{image_id}{extension}"
        file_path = collection_path / filename
        
        # 保存图片文件
        try:
            with open(file_path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            raise RuntimeError(f"保存图片文件失败: {str(e)}") from e
        
        # 记录索引信息
        image_info = {
            "image_id": image_id,
            "file_path": str(file_path),
            "relative_path": f"{collection_name}/{filename}",
            "collection_name": collection_name,
            "mime_type": mime_type,
            **(metadata or {})
        }
        
        self._image_index[image_id] = image_info
        
        return str(file_path)
    
    def save_image_from_base64(
        self,
        image_id: str,
        base64_data: str,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None
    ) -> str:
        """
        从 Base64 编码的字符串保存图片
        
        Args:
            image_id: 图片唯一标识符
            base64_data: Base64 编码的图片数据（可以包含 data URI 前缀）
            collection_name: 集合名称
            metadata: 图片元数据（可选）
            trace: 追踪上下文（可选）
        
        Returns:
            str: 保存后的图片文件路径
        """
        # 移除 data URI 前缀（如果存在）
        if base64_data.startswith("data:"):
            base64_data = base64_data.split(",", 1)[1]
        
        # 解码 Base64
        try:
            image_data = b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Base64 解码失败: {str(e)}") from e
        
        return self.save_image(image_id, image_data, collection_name, metadata, trace)
    
    def get_image_path(self, image_id: str) -> Optional[str]:
        """
        根据 image_id 获取图片文件路径
        
        Args:
            image_id: 图片唯一标识符
        
        Returns:
            Optional[str]: 图片文件路径，如果不存在则返回 None
        """
        if image_id in self._image_index:
            return self._image_index[image_id]["file_path"]
        return None
    
    def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 image_id 获取图片完整信息
        
        Args:
            image_id: 图片唯一标识符
        
        Returns:
            Optional[Dict[str, Any]]: 图片信息字典，如果不存在则返回 None
        """
        return self._image_index.get(image_id)
    
    def load_index(self, collection_name: str) -> None:
        """
        从文件系统加载索引
        
        Args:
            collection_name: 集合名称
        
        Raises:
            FileNotFoundError: 当索引文件不存在时
            ValueError: 当索引文件格式不正确时
        """
        if not collection_name:
            raise ValueError("collection_name 不能为空")
        
        index_file = self._base_path / collection_name / "index.json"
        
        if not index_file.exists():
            # 索引文件不存在，使用空索引
            self._image_index = {}
            return
        
        # 加载 JSON
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        except Exception as e:
            raise ValueError(f"加载索引文件失败: {str(e)}") from e
        
        # 验证数据格式
        if not isinstance(index_data, dict) or "images" not in index_data:
            raise ValueError(f"索引文件格式不正确: {index_file}")
        
        # 恢复索引
        self._image_index = index_data.get("images", {})
    
    def save_index(self, collection_name: str) -> None:
        """
        将索引保存到文件系统
        
        Args:
            collection_name: 集合名称
        
        Raises:
            ValueError: 当 collection_name 为空时
            RuntimeError: 当保存失败时
        """
        if not collection_name:
            raise ValueError("collection_name 不能为空")
        
        collection_path = self._base_path / collection_name
        collection_path.mkdir(parents=True, exist_ok=True)
        
        index_file = collection_path / "index.json"
        
        # 准备保存的数据
        index_data = {
            "collection_name": collection_name,
            "images": self._image_index,
            "total_images": len(self._image_index)
        }
        
        # 保存为 JSON
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"保存索引文件失败: {str(e)}") from e
    
    def _get_extension_from_mime_type(self, mime_type: str) -> str:
        """
        根据 MIME 类型获取文件扩展名
        
        Args:
            mime_type: MIME 类型，例如 "image/png", "image/jpeg"
        
        Returns:
            str: 文件扩展名，例如 ".png", ".jpg"
        """
        mime_to_ext = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/svg+xml": ".svg",
        }
        
        return mime_to_ext.get(mime_type.lower(), ".png")
    
    def get_total_images(self) -> int:
        """
        获取当前索引中的图片总数
        
        Returns:
            int: 图片总数
        """
        return len(self._image_index)
    
    def list_images(self, collection_name: Optional[str] = None) -> List[str]:
        """
        列出所有图片 ID（可选按集合过滤）
        
        Args:
            collection_name: 集合名称（可选），如果提供则只返回该集合的图片
        
        Returns:
            List[str]: 图片 ID 列表
        """
        if collection_name:
            return [
                image_id for image_id, info in self._image_index.items()
                if info.get("collection_name") == collection_name
            ]
        return list(self._image_index.keys())
