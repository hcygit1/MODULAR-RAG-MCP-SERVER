#!/usr/bin/env python3
"""
使用 MinerU 云端 API 解析 PDF 并入库（CLI 入口）

与 ingest_document_mineru MCP 工具等价，可直接从命令行调用。
"""
import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="使用 MinerU 云端 API 解析 PDF 并入库",
        epilog="""
示例:
  uv run python scripts/ingest_mineru.py tests/fixtures/sample_documents/sample2.pdf
  uv run python scripts/ingest_mineru.py --collection report /path/to/document.pdf
"""
    )
    parser.add_argument("file_path", type=str, help="PDF 文件路径")
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="目标集合名，默认使用配置中的 vector_store.collection_name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()

    file_path = Path(args.file_path).resolve()
    if not file_path.exists() or not file_path.is_file():
        logger.error("文件不存在: %s", file_path)
        sys.exit(1)

    from src.core.settings import load_settings
    from src.libs.loader.mineru_cloud_client import MinerUCloudClient
    from src.libs.loader.mineru_result_adapter import to_document
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings(args.config)
    collection_name = (args.collection or "").strip() or settings.vector_store.collection_name

    token = getattr(getattr(settings, "mineru", None), "api_token", "") or ""
    if not token:
        import os
        token = os.environ.get("MINERU_API_TOKEN", "")
    if not token or not str(token).strip():
        logger.error("MinerU API Token 未配置，请在 config/settings.local.yaml 的 mineru.api_token 中填写")
        sys.exit(1)

    logger.info("使用 MinerU 解析: %s -> 集合 %s", file_path, collection_name)
    client = MinerUCloudClient(
        api_token=token.strip(),
        model_version=settings.mineru.model_version,
        poll_interval_seconds=settings.mineru.poll_interval_seconds,
        poll_timeout_seconds=settings.mineru.poll_timeout_seconds,
    )

    try:
        raw = client.upload_and_parse(str(file_path))
        document = to_document(raw)
        pipeline = IngestionPipeline(settings)
        try:
            chunk_count = pipeline.process_document(document, collection_name)
            logger.info("✅ MinerU 解析并入库成功，共 %d 个 chunks 到集合 %s", chunk_count, collection_name)
        finally:
            pipeline.close()
    except Exception as e:
        logger.exception("MinerU 解析或入库失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
