"""MinerUCloudClient 单元测试（Mock HTTP）"""
import json
import zipfile
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest

from src.libs.loader.mineru_cloud_client import MinerUCloudClient, MinerURawResult


@pytest.fixture
def mock_batch_response():
    """模拟 file-urls/batch 响应"""
    return {
        "code": 0,
        "data": {
            "batch_id": "test-batch-123",
            "file_urls": ["https://mock-upload.example.com/upload/xxx"],
        },
        "msg": "ok",
    }


@pytest.fixture
def mock_extract_response_done():
    """模拟 extract-results/batch 完成响应"""
    return {
        "code": 0,
        "data": {
            "batch_id": "test-batch-123",
            "extract_result": [
                {
                    "file_name": "test.pdf",
                    "state": "done",
                    "full_zip_url": "https://mock-cdn.example.com/result.zip",
                    "err_msg": "",
                }
            ],
        },
        "msg": "ok",
    }


def _make_fake_zip_with_md(md_content: str = "# Test\n\nContent.") -> bytes:
    """构造包含 md 的 zip"""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("auto/parsed_content.md", md_content.encode("utf-8"))
    return buf.getvalue()


@patch("src.libs.loader.mineru_cloud_client.requests")
def test_upload_and_parse_success(mock_requests, mock_batch_response, mock_extract_response_done, tmp_path):
    """完整流程 Mock 成功"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    mock_requests.post.return_value = MagicMock(
        status_code=200,
        json=lambda: mock_batch_response,
        raise_for_status=MagicMock(),
    )
    mock_requests.put.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
    mock_requests.get.side_effect = [
        MagicMock(status_code=200, json=lambda: mock_extract_response_done, raise_for_status=MagicMock()),
        MagicMock(
            status_code=200,
            content=_make_fake_zip_with_md(),
            raise_for_status=MagicMock(),
        ),
    ]

    client = MinerUCloudClient(api_token="test-token")
    result = client.upload_and_parse(str(pdf_path))

    assert isinstance(result, MinerURawResult)
    assert result.markdown_text == "# Test\n\nContent."
    assert result.source_path == str(pdf_path.resolve())
    assert result.images == []
    assert result.err_msg is None


def test_client_requires_token():
    """api_token 为空时抛出 ValueError"""
    with pytest.raises(ValueError, match="api_token"):
        MinerUCloudClient(api_token="")


def test_client_requires_token_not_whitespace():
    """api_token 仅空白时抛出 ValueError"""
    with pytest.raises(ValueError, match="api_token"):
        MinerUCloudClient(api_token="   ")
