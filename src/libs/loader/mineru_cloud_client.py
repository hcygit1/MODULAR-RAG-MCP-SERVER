"""
MinerU 云端 API 客户端

封装 mineru.net 云端 API：申请上传链接 → 上传文件 → 轮询结果 → 下载解压。
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class MinerURawResult:
    """MinerU 原始解析结果，供 Adapter 转换为 Document"""

    markdown_text: str
    source_path: str
    # (img_path_in_zip, page_idx, image_bytes)
    images: List[Tuple[str, int, bytes]]
    content_list: Optional[List[Dict[str, Any]]] = None
    err_msg: Optional[str] = None


class MinerUCloudClient:
    """
    MinerU 云端 API 客户端

    流程：申请上传链接 → PUT 上传 → 轮询 batch 结果 → 下载 zip → 解压读取 md、images。
    """

    BASE_URL = "https://mineru.net/api/v4"

    def __init__(
        self,
        api_token: str,
        model_version: str = "vlm",
        poll_interval_seconds: int = 5,
        poll_timeout_seconds: int = 600,
        request_timeout: int = 60,
    ):
        """
        初始化 MinerU 云端客户端

        Args:
            api_token: MinerU API Token（从 mineru.net 申请）
            model_version: 模型版本，pipeline | vlm | MinerU-HTML
            poll_interval_seconds: 轮询间隔（秒）
            poll_timeout_seconds: 轮询超时（秒）
            request_timeout: 单次 HTTP 请求超时（秒）
        """
        if not api_token or not str(api_token).strip():
            raise ValueError("MinerU api_token 不能为空")
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests 未安装，请安装: pip install requests")

        self._api_token = api_token.strip()
        self._model_version = model_version
        self._poll_interval = poll_interval_seconds
        self._poll_timeout = poll_timeout_seconds
        self._request_timeout = request_timeout
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_token}",
            "Accept": "*/*",
        }

    def upload_and_parse(
        self,
        file_path: str,
        trace: Optional[Any] = None,
    ) -> MinerURawResult:
        """
        上传本地文件到 MinerU 云端并解析，返回原始结果

        Args:
            file_path: 本地 PDF 文件路径
            trace: 追踪上下文（可选）

        Returns:
            MinerURawResult: 解析结果

        Raises:
            FileNotFoundError: 文件不存在
            RuntimeError: API 调用失败、轮询超时、解析失败
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_name = path.name
        file_ext = path.suffix.lower()
        if file_ext not in (".pdf", ".doc", ".docx", ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".html"):
            raise ValueError(f"不支持的文件类型: {file_ext}")

        # 1. 申请上传链接
        batch_id, file_urls = self._request_upload_urls(file_name)
        if not file_urls:
            raise RuntimeError("未获取到上传链接")

        # 2. 上传文件
        with open(path, "rb") as f:
            self._upload_file(file_urls[0], f.read())

        # 3. 轮询结果
        full_zip_url = self._poll_until_done(batch_id, file_name)
        if not full_zip_url:
            raise RuntimeError("解析超时或失败")

        # 4. 下载并解析 zip
        return self._download_and_parse_zip(full_zip_url, str(path.resolve()))

    def _request_upload_urls(self, file_name: str) -> Tuple[str, List[str]]:
        """申请上传链接，返回 (batch_id, file_urls)"""
        url = f"{self.BASE_URL}/file-urls/batch"
        payload = {
            "files": [{"name": file_name, "data_id": f"ingest_{file_name}"}],
            "model_version": self._model_version,
        }

        resp = requests.post(
            url,
            headers=self._headers,
            json=payload,
            timeout=self._request_timeout,
        )
        resp.raise_for_status()

        data = resp.json()
        if data.get("code") != 0:
            msg = data.get("msg", "未知错误")
            raise RuntimeError(f"MinerU 申请上传链接失败: {msg}")

        result = data.get("data", {})
        batch_id = result.get("batch_id")
        file_urls = result.get("file_urls") or result.get("files") or []
        if not batch_id:
            raise RuntimeError("MinerU 响应缺少 batch_id")
        if isinstance(file_urls, list) and file_urls and isinstance(file_urls[0], str):
            pass
        else:
            file_urls = []

        return str(batch_id), file_urls

    def _upload_file(self, upload_url: str, data: bytes) -> None:
        """PUT 上传文件，不设置 Content-Type"""
        resp = requests.put(
            upload_url,
            data=data,
            timeout=self._request_timeout,
        )
        resp.raise_for_status()
        if resp.status_code != 200:
            raise RuntimeError(f"上传失败: HTTP {resp.status_code}")

    def _poll_until_done(self, batch_id: str, file_name: str) -> Optional[str]:
        """轮询 batch 结果，返回 full_zip_url 或 None"""
        url = f"{self.BASE_URL}/extract-results/batch/{batch_id}"
        deadline = time.time() + self._poll_timeout

        while time.time() < deadline:
            resp = requests.get(url, headers=self._headers, timeout=self._request_timeout)
            resp.raise_for_status()

            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(f"MinerU 查询结果失败: {data.get('msg', '未知错误')}")

            result = data.get("data", {})
            extract_result = result.get("extract_result") or result.get("extract_results")
            if not extract_result:
                extract_result = []

            for item in extract_result if isinstance(extract_result, list) else [extract_result]:
                state = item.get("state", "")
                if state == "failed":
                    err = item.get("err_msg", "解析失败")
                    raise RuntimeError(f"MinerU 解析失败: {err}")
                if state == "done":
                    zip_url = item.get("full_zip_url")
                    if zip_url:
                        return zip_url

            time.sleep(self._poll_interval)

        return None

    def _download_and_parse_zip(self, zip_url: str, source_path: str) -> MinerURawResult:
        """下载 zip，解压，读取 md 和 images"""
        resp = requests.get(zip_url, timeout=self._request_timeout)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
            with tempfile.TemporaryDirectory(prefix="mineru_") as tmpdir:
                zf.extractall(tmpdir)

                # 查找 md 文件
                markdown_text = self._find_and_read_md(tmpdir)
                if not markdown_text:
                    markdown_text = ""

                # 收集 images
                images: List[Tuple[str, int, bytes]] = []
                content_list = self._find_content_list(tmpdir)

                for root, _dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                            rel_path = os.path.relpath(os.path.join(root, f), tmpdir)
                            # 标准化为正斜杠，与 md 中的引用一致
                            rel_path = rel_path.replace("\\", "/")
                            with open(os.path.join(root, f), "rb") as imgf:
                                img_bytes = imgf.read()
                            page_idx = self._infer_page_from_content_list(rel_path, content_list)
                            images.append((rel_path, page_idx, img_bytes))

                # 按 content_list 顺序或 path 排序
                images.sort(key=lambda x: (x[1], x[0]))

        return MinerURawResult(
            markdown_text=markdown_text,
            source_path=source_path,
            images=images,
            content_list=content_list,
        )

    def _find_and_read_md(self, root: str) -> str:
        """递归查找第一个 .md 文件并读取"""
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".md"):
                    path = os.path.join(dirpath, f)
                    try:
                        with open(path, "r", encoding="utf-8", errors="replace") as fp:
                            return fp.read()
                    except Exception as e:
                        logger.warning("读取 md 文件失败 %s: %s", path, e)
        return ""

    def _find_content_list(self, root: str) -> Optional[List[Dict[str, Any]]]:
        """查找 content_list.json"""
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if "content_list" in f.lower() and f.lower().endswith(".json"):
                    path = os.path.join(dirpath, f)
                    try:
                        with open(path, "r", encoding="utf-8") as fp:
                            data = json.load(fp)
                            return data if isinstance(data, list) else None
                    except Exception as e:
                        logger.debug("读取 content_list 失败 %s: %s", path, e)
        return None

    def _infer_page_from_content_list(
        self,
        img_rel_path: str,
        content_list: Optional[List[Any]],
    ) -> int:
        """从 content_list 推断图片的 page_idx。支持扁平 [dict,...] 或嵌套 [[dict,...],...] 结构。"""
        if not content_list:
            return 0

        def _search_items(items: list, page_idx: int = 0) -> Optional[int]:
            for item in items:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        path = item.get("img_path", "") or item.get("image_path", "")
                        if path and (img_rel_path in path or path in img_rel_path):
                            return item.get("page_idx", page_idx)
                elif isinstance(item, list):
                    found = _search_items(item, page_idx)
                    if found is not None:
                        return found
            return None

        # 嵌套结构：[[{...}, {...}], [...]]，外层索引为 page_idx
        if content_list and isinstance(content_list[0], list):
            for page_idx, page_items in enumerate(content_list):
                found = _search_items(page_items, page_idx)
                if found is not None:
                    return found
        else:
            found = _search_items(content_list)
            if found is not None:
                return found
        return 0
