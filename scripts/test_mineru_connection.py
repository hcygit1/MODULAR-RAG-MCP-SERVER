#!/usr/bin/env python3
"""
测试 MinerU 云端 API 连接与密钥有效性

从 config/settings.yaml（含 settings.local.yaml 合并）读取 mineru.api_token，
调用 MinerU API 申请上传链接，验证密钥是否正确。
"""
import sys
from pathlib import Path

# 项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import requests
except ImportError:
    print("❌ 缺少 requests，请运行: pip install requests")
    sys.exit(1)


def main():
    from src.core.settings import load_settings

    print("正在加载配置 (config/settings.yaml + settings.local.yaml)...")
    try:
        settings = load_settings("config/settings.yaml")
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        sys.exit(1)

    token = getattr(getattr(settings, "mineru", None), "api_token", "") or ""
    if not token:
        token = __import__("os").environ.get("MINERU_API_TOKEN", "")
    if not token or not str(token).strip():
        print("❌ MinerU api_token 未配置")
        print("   请在 config/settings.local.yaml 的 mineru.api_token 中填写，或设置环境变量 MINERU_API_TOKEN")
        sys.exit(1)

    print(f"已读取 api_token（前 8 位: {token[:8]}***）")
    print("正在请求 MinerU API (POST /file-urls/batch)...")

    url = "https://mineru.net/api/v4/file-urls/batch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token.strip()}",
        "Accept": "*/*",
    }
    payload = {
        "files": [{"name": "test.pdf", "data_id": "ingest_test.pdf"}],
        "model_version": "vlm",
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"❌ 网络请求失败: {e}")
        sys.exit(1)

    # HTTP 错误
    if resp.status_code == 401:
        print("❌ 密钥无效：认证失败 (401 Unauthorized)")
        sys.exit(1)
    if resp.status_code == 403:
        print("❌ 密钥无效：无权限 (403 Forbidden)")
        sys.exit(1)
    if resp.status_code != 200:
        print(f"❌ 请求失败: HTTP {resp.status_code}")
        try:
            print(f"   响应: {resp.text[:200]}")
        except Exception:
            pass
        sys.exit(1)

    # 解析 JSON
    try:
        data = resp.json()
    except Exception:
        print("❌ 响应非 JSON 格式")
        sys.exit(1)

    if data.get("code") != 0:
        msg = data.get("msg", "未知错误")
        print(f"❌ MinerU 返回错误: {msg}")
        if "token" in msg.lower() or "auth" in msg.lower() or "invalid" in msg.lower():
            print("   可能原因：密钥无效或已过期")
        sys.exit(1)

    result = data.get("data", {})
    batch_id = result.get("batch_id")
    file_urls = result.get("file_urls") or result.get("files") or []

    if not batch_id or not file_urls:
        print("❌ 响应格式异常，未获取到 batch_id 或 file_urls")
        sys.exit(1)

    print("✅ 连接成功！密钥有效。")
    print(f"   batch_id: {batch_id}")
    print(f"   已获取上传链接，可正常使用 ingest_document_mineru 工具")


if __name__ == "__main__":
    main()
