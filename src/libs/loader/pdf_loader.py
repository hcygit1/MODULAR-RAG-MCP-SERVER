"""
PDF Loader 实现

使用 MarkItDown 将 PDF 文件转换为 Markdown 格式的 Document 对象。
支持使用 PyMuPDF 提取图片数据和位置信息。
"""
import os
import hashlib
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

from src.ingestion.models import Document
from src.libs.loader.base_loader import BaseLoader

# 尝试导入 MarkItDown
try:
    from markitdown import MarkItDown  # type: ignore
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    MarkItDown = None

# 尝试导入 PyMuPDF（可选依赖，用于图片提取）
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None


class PdfLoader(BaseLoader):
    """
    PDF Loader 实现
    
    使用 MarkItDown 将 PDF 文件转换为 Markdown 格式。
    支持提取文本、图片引用和基础元数据。
    如果安装了 PyMuPDF，会提取图片并在 Markdown 中插入占位符。
    """
    
    def __init__(self):
        """初始化 PDF Loader"""
        if not MARKITDOWN_AVAILABLE:
            raise RuntimeError(
                "MarkItDown 未安装。请安装: pip install markitdown"
            )
        self._md = MarkItDown()
        self._supported_extensions = [".pdf"]
    
    def load(self, path: str, trace: Optional[Any] = None) -> Document:
        """
        加载 PDF 文件并转换为 Document 对象
        
        Args:
            path: PDF 文件路径
            trace: 追踪上下文（可选）
        
        Returns:
            Document: 包含 Markdown 文本和元数据的文档对象
        
        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当文件路径无效或不是 PDF 文件时
            RuntimeError: 当 PDF 解析失败时
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        
        # 验证文件扩展名
        file_ext = Path(path).suffix.lower()
        if file_ext not in self._supported_extensions:
            raise ValueError(
                f"不支持的文件类型: {file_ext}。"
                f"支持的扩展名: {self._supported_extensions}"
            )
        
        try:
            # 1. 使用 MarkItDown 转换 PDF 为 Markdown
            result = self._md.convert(path)
            
            # 提取文本内容
            markdown_text = result.text_content if hasattr(result, 'text_content') else str(result)
            
            # 如果没有文本内容，使用空字符串
            if not markdown_text or not markdown_text.strip():
                markdown_text = ""
            
            # 生成文档 ID（基于文件路径的哈希）
            doc_id = self._generate_doc_id(path)
            
            # 2. 提取图片（如果 PyMuPDF 可用）
            image_data_dict = {}  # {image_id: bytes}
            image_metadata_list = []  # [{image_id, page, y_position, ...}]
            page_images = {}  # {page_num: [(image_id, y_position, bytes, metadata), ...]}
            
            if PYMUPDF_AVAILABLE:
                try:
                    image_data_dict, image_metadata_list, page_images = \
                        self._extract_images_from_pdf(path, doc_id)
                    
                    # 3. 在 Markdown 中插入图片占位符
                    if page_images:
                        markdown_text = self._insert_image_placeholders(
                            markdown_text,
                            page_images,
                            path
                        )
                except Exception as e:
                    # 图片提取失败不影响主流程，记录警告
                    # TODO: 使用 logging 记录警告
                    print(f"警告: 图片提取失败: {str(e)}")
            
            # 4. 构建元数据
            metadata = self._build_metadata(path, result)
            
            # 5. 保存图片数据到元数据（如果提取了图片）
            if image_data_dict:
                metadata["image_data"] = image_data_dict  # 图片二进制数据
            if image_metadata_list:
                metadata["images"] = image_metadata_list  # 图片元数据列表
            
            return Document(
                id=doc_id,
                text=markdown_text,
                metadata=metadata
            )
            
        except Exception as e:
            raise RuntimeError(
                f"PDF 解析失败: {path}。错误: {str(e)}"
            ) from e
    
    def _generate_doc_id(self, path: str) -> str:
        """
        生成文档唯一标识符
        
        基于文件路径生成稳定的文档 ID。
        
        Args:
            path: 文件路径
        
        Returns:
            str: 文档 ID
        """
        # 使用绝对路径确保唯一性
        abs_path = os.path.abspath(path)
        # 使用 SHA256 哈希的前 16 个字符作为 ID
        hash_obj = hashlib.sha256(abs_path.encode('utf-8'))
        return f"doc_{hash_obj.hexdigest()[:16]}"
    
    def _build_metadata(
        self,
        path: str,
        result: Any
    ) -> Dict[str, Any]:
        """
        构建文档元数据
        
        Args:
            path: 文件路径
            result: MarkItDown 转换结果
        
        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata: Dict[str, Any] = {
            "source_path": os.path.abspath(path),
            "doc_type": "pdf",
        }
        
        # 提取文件名（不含扩展名）作为可能的标题
        file_name = Path(path).stem
        if file_name:
            metadata["title"] = file_name
        
        # 尝试提取图片引用（如果 MarkItDown 支持）
        # 注意：如果 PyMuPDF 提取了图片，这些会被覆盖为 PyMuPDF 的数据
        # 保留 MarkItDown 的原始数据为 markitdown_images（如果存在）
        if hasattr(result, 'images') and result.images:
            metadata["markitdown_images"] = result.images
        elif hasattr(result, 'image_refs') and result.image_refs:
            metadata["markitdown_images"] = result.image_refs
        
        # 如果 PyMuPDF 没有提取图片，初始化空列表
        # 如果 PyMuPDF 提取了图片，会在 load() 方法中被覆盖
        if "images" not in metadata:
            metadata["images"] = []
        
        # 尝试提取其他元数据
        if hasattr(result, 'metadata') and result.metadata:
            # 合并额外的元数据
            for key, value in result.metadata.items():
                if key not in metadata:
                    metadata[key] = value
        
        return metadata
    
    def _extract_images_from_pdf(
        self,
        pdf_path: str,
        doc_id: str
    ) -> Tuple[Dict[str, bytes], List[Dict[str, Any]], Dict[int, List[Tuple[str, float, bytes, Dict[str, Any]]]]]:
        """
        从 PDF 中提取图片数据和位置信息
        
        Args:
            pdf_path: PDF 文件路径
            doc_id: 文档 ID，用于生成图片 ID
        
        Returns:
            Tuple[Dict[str, bytes], List[Dict[str, Any]], Dict[int, List[Tuple]]]:
            - image_data_dict: {image_id: bytes} 图片二进制数据字典
            - image_metadata_list: [{image_id, page, y_position, ...}] 图片元数据列表
            - page_images: {page_num: [(image_id, y_position, bytes, metadata), ...]} 按页组织的图片列表
        
        Raises:
            RuntimeError: 当 PyMuPDF 不可用时
        """
        if not PYMUPDF_AVAILABLE:
            return {}, [], {}
        
        image_data_dict = {}  # {image_id: bytes}
        image_metadata_list = []  # [{image_id, page, y_position, ...}]
        page_images = {}  # {page_num: [(image_id, y_position, bytes, metadata), ...]}
        
        # 打开 PDF 文件
        doc = fitz.open(pdf_path)
        
        try:
            # 遍历每一页
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 获取页面中的所有图片
                # get_images(full=True) 返回: [(xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter, referencer), ...]
                image_list = page.get_images(full=True)
                
                page_images[page_num] = []
                
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]  # 图片的 xref（交叉引用）
                    
                    try:
                        # 提取图片数据
                        # extract_image() 返回: {
                        #   "width": int,
                        #   "height": int,
                        #   "colorspace": int,
                        #   "bpc": int,
                        #   "image": bytes,  # 图片二进制数据
                        #   "ext": str,      # 扩展名，如 "png", "jpeg"
                        #   "smask": int,    # 可选：软遮罩
                        #   "xres": int,     # X 分辨率
                        #   "yres": int,     # Y 分辨率
                        # }
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        ext = base_image.get("ext", "png")
                        
                        # 生成唯一的 image_id
                        image_id = f"{doc_id}_page_{page_num}_img_{img_idx}"
                        
                        # 获取图片在页面中的位置（Y 坐标）
                        try:
                            # 获取图片在页面中的显示位置
                            # get_image_rects() 返回: [fitz.Rect(x0, y0, x1, y1), ...]
                            image_rects = page.get_image_rects(xref)
                            if image_rects:
                                # 使用第一个矩形的位置
                                y_position = image_rects[0].y0  # 图片上边缘的 Y 坐标
                                bbox = {
                                    "x0": image_rects[0].x0,
                                    "y0": image_rects[0].y0,
                                    "x1": image_rects[0].x1,
                                    "y1": image_rects[0].y1,
                                }
                            else:
                                # 如果无法获取位置，使用索引作为排序依据
                                y_position = img_idx * 1000
                                bbox = None
                        except Exception:
                            # 如果获取位置失败，使用索引
                            y_position = img_idx * 1000
                            bbox = None
                        
                        # 保存图片数据
                        image_data_dict[image_id] = image_bytes
                        
                        # 构建图片元数据
                        img_metadata = {
                            "image_id": image_id,
                            "page": page_num,
                            "xref": xref,
                            "y_position": y_position,
                            "width": base_image.get("width", 0),
                            "height": base_image.get("height", 0),
                            "mime_type": f"image/{ext}",
                            "ext": ext,
                            "bbox": bbox,
                        }
                        
                        image_metadata_list.append(img_metadata)
                        
                        # 保存到页面图片列表（按 Y 坐标排序）
                        page_images[page_num].append((
                            image_id,
                            y_position,
                            image_bytes,
                            img_metadata
                        ))
                        
                    except Exception as e:
                        # 单个图片提取失败，记录但继续处理其他图片
                        # TODO: 使用 logging 记录警告
                        print(f"警告: 提取第 {page_num} 页第 {img_idx} 张图片失败: {str(e)}")
                        continue
                
                # 对每页的图片按 Y 坐标排序（从上到下）
                page_images[page_num].sort(key=lambda x: x[1])
        
        finally:
            doc.close()
        
        return image_data_dict, image_metadata_list, page_images
    
    def _map_pages_to_markdown(
        self,
        pdf_path: str,
        markdown_text: str,
        fingerprint_length: int = 200
    ) -> Dict[int, Tuple[int, int]]:
        """
        建立 PDF 页面到 Markdown 行的映射
        
        通过提取每页文本的"指纹"（前N个字符），在 Markdown 中查找匹配位置，
        建立页面到 Markdown 行的映射关系。
        
        Args:
            pdf_path: PDF 文件路径
            markdown_text: Markdown 文本
            fingerprint_length: 文本指纹长度（默认200个字符）
        
        Returns:
            Dict[int, Tuple[int, int]]: {page_num: (start_line, end_line)}
            - start_line: 该页在 Markdown 中的起始行号（从0开始）
            - end_line: 该页在 Markdown 中的结束行号（不包含）
        
        注意：
            - 如果找不到匹配，该页不会出现在返回的字典中
            - end_line 可能是下一页的起始行，或文档末尾的行号
        """
        if not PYMUPDF_AVAILABLE:
            return {}
        
        page_to_markdown = {}  # {page_num: (start_line, end_line)}
        markdown_lines = markdown_text.split('\n')
        
        # 打开 PDF 文件
        doc = fitz.open(pdf_path)
        
        try:
            # 提取每页的文本指纹
            page_fingerprints = {}  # {page_num: fingerprint}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                # 提取文本指纹（前N个字符）
                fingerprint = page_text[:fingerprint_length].strip() if page_text else ""
                if fingerprint:
                    page_fingerprints[page_num] = fingerprint
            
            # 在 Markdown 中查找每页的位置
            for page_num in sorted(page_fingerprints.keys()):
                fingerprint = page_fingerprints[page_num]
                fingerprint_lower = fingerprint.lower()
                
                # 在 Markdown 中查找匹配位置
                page_start_line = None
                
                # 从上一页的结束位置开始查找（如果存在），避免重复匹配
                search_start = 0
                if page_num > 0 and (page_num - 1) in page_to_markdown:
                    search_start = page_to_markdown[page_num - 1][1]
                
                for line_idx in range(search_start, len(markdown_lines)):
                    line = markdown_lines[line_idx]
                    # 检查指纹是否出现在这一行（不区分大小写）
                    if fingerprint_lower and fingerprint_lower in line.lower():
                        page_start_line = line_idx
                        break
                
                if page_start_line is None:
                    # 如果找不到匹配，跳过该页
                    continue
                
                # 查找该页的结束位置（下一页的起始位置）
                page_end_line = len(markdown_lines)  # 默认到文档末尾
                
                # 查找下一页的起始位置
                for next_page_num in range(page_num + 1, len(doc)):
                    if next_page_num in page_fingerprints:
                        next_fingerprint = page_fingerprints[next_page_num]
                        next_fingerprint_lower = next_fingerprint.lower()
                        
                        # 从当前页之后开始查找
                        for next_line_idx in range(page_start_line + 1, len(markdown_lines)):
                            if next_fingerprint_lower in markdown_lines[next_line_idx].lower():
                                page_end_line = next_line_idx
                                break
                        
                        if page_end_line < len(markdown_lines):
                            break
                
                # 记录映射
                page_to_markdown[page_num] = (page_start_line, page_end_line)
        
        finally:
            doc.close()
        
        return page_to_markdown
    
    def _insert_image_placeholders_simple(
        self,
        markdown_text: str,
        page_images: Dict[int, List[Tuple[str, float, bytes, Dict[str, Any]]]]
    ) -> str:
        """
        简单策略：在文档末尾追加所有图片占位符
        
        作为回退策略，当精确插入失败时使用。
        
        Args:
            markdown_text: 原始 Markdown 文本
            page_images: {page_num: [(image_id, y_position, bytes, metadata), ...]}
        
        Returns:
            str: 插入占位符后的 Markdown 文本
        """
        if not page_images:
            return markdown_text
        
        # 收集所有图片占位符
        all_placeholders = []
        for page_num in sorted(page_images.keys()):
            for image_id, _, _, _ in page_images[page_num]:
                all_placeholders.append(f"[IMAGE: {image_id}]")
        
        if all_placeholders:
            # 在文档末尾追加所有占位符
            return markdown_text + '\n\n' + '\n'.join(all_placeholders)
        
        return markdown_text
    
    def _insert_image_placeholders(
        self,
        markdown_text: str,
        page_images: Dict[int, List[Tuple[str, float, bytes, Dict[str, Any]]]],
        pdf_path: str,
        page_to_markdown: Optional[Dict[int, Tuple[int, int]]] = None
    ) -> str:
        """
        根据图片 Y 坐标的相对位置精确插入占位符
        
        策略：
        1. 找到每页在 Markdown 中的起始和结束行
        2. 获取图片在 PDF 页面中的 Y 坐标
        3. 计算图片的相对位置（Y坐标 / 页面高度）
        4. 在 Markdown 对应相对位置插入占位符
        
        Args:
            markdown_text: 原始 Markdown 文本
            page_images: {page_num: [(image_id, y_position, bytes, metadata), ...]}
            pdf_path: PDF 文件路径（用于获取页面高度）
            page_to_markdown: 页面到 Markdown 行的映射（可选，如果未提供则自动计算）
        
        Returns:
            str: 插入占位符后的 Markdown 文本
        """
        if not page_images or not PYMUPDF_AVAILABLE:
            return self._insert_image_placeholders_simple(markdown_text, page_images)
        
        # 如果没有提供页面映射，尝试计算
        if page_to_markdown is None:
            try:
                page_to_markdown = self._map_pages_to_markdown(pdf_path, markdown_text)
            except Exception as e:
                # 如果映射失败，使用简单策略
                # TODO: 使用 logging 记录警告
                print(f"警告: 页面映射失败，使用简单策略: {str(e)}")
                return self._insert_image_placeholders_simple(markdown_text, page_images)
        
        # 如果没有找到任何页面映射，使用简单策略
        if not page_to_markdown:
            return self._insert_image_placeholders_simple(markdown_text, page_images)
        
        markdown_lines = markdown_text.split('\n')
        insertions = []  # [(line_index, placeholder), ...]
        
        # 打开 PDF 获取页面高度
        doc = fitz.open(pdf_path)
        
        try:
            # 为每页的图片计算插入位置
            for page_num in sorted(page_images.keys(), reverse=True):
                if page_num not in page_to_markdown:
                    # 如果找不到页面映射，在该页的图片使用简单策略（追加到末尾）
                    for image_id, _, _, _ in page_images[page_num]:
                        insertions.append((len(markdown_lines), f"[IMAGE: {image_id}]"))
                    continue
                
                start_line, end_line = page_to_markdown[page_num]
                images = page_images[page_num]
                
                if not images:
                    continue
                
                # 获取页面高度（用于计算相对位置）
                try:
                    page = doc[page_num]
                    page_height = page.rect.height  # 页面高度（点）
                except Exception:
                    # 如果获取失败，使用默认值
                    page_height = 800  # 默认 A4 页面高度（点）
                
                # 计算该页在 Markdown 中的行数
                page_lines_count = end_line - start_line
                
                if page_lines_count <= 0:
                    # 如果该页没有文本行，在末尾插入所有图片
                    for image_id, _, _, _ in images:
                        insertions.append((end_line, f"[IMAGE: {image_id}]"))
                    continue
                
                # 按 Y 坐标排序图片（从上到下）
                images_sorted = sorted(images, key=lambda x: x[1])
                
                # 根据图片 Y 坐标计算相对位置并插入
                for image_id, img_y_pos, _, img_meta in images_sorted:
                    # 计算图片在页面中的相对位置（0.0 到 1.0）
                    relative_position = img_y_pos / page_height if page_height > 0 else 0.5
                    
                    # 限制在合理范围内
                    relative_position = max(0.0, min(1.0, relative_position))
                    
                    # 计算在 Markdown 中的对应行号
                    target_line = start_line + int(relative_position * page_lines_count)
                    
                    # 确保不超出页面范围
                    target_line = max(start_line, min(target_line, end_line - 1))
                    
                    insertions.append((target_line, f"[IMAGE: {image_id}]"))
        
        finally:
            doc.close()
        
        # 按行号倒序插入（避免插入后索引变化）
        insertions.sort(key=lambda x: x[0], reverse=True)
        
        # 执行插入
        for line_idx, placeholder in insertions:
            markdown_lines.insert(line_idx, placeholder)
        
        return '\n'.join(markdown_lines)
    
    def get_supported_extensions(self) -> list[str]:
        """
        获取支持的文件扩展名列表
        
        Returns:
            list[str]: 支持的文件扩展名列表
        """
        return self._supported_extensions.copy()
