#!/usr/bin/env python3
"""
数据摄取脚本入口

支持离线批量处理文档，将文档转换为向量和索引存储。
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def collect_pdf_files(path: Path) -> list[Path]:
    """Return each PDF below ``path`` once, in deterministic order."""
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"不支持的文件类型: {path.suffix}（当前仅支持 PDF）")
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.pdf"))
    raise ValueError(f"无效的路径类型: {path}")


def main(argv: Sequence[str] | None = None) -> None:
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="数据摄取脚本：将文档转换为向量和索引存储",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件
  modular-rag-mcp-server ingest --collection my_collection --path document.pdf

  # 强制重新处理（忽略完整性检查）
  modular-rag-mcp-server ingest --collection my_collection --path document.pdf --force

  # 处理目录下的所有 PDF 文件
  modular-rag-mcp-server ingest --collection my_collection --path ./documents/
        """
    )

    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="集合名称，用于组织存储的数据"
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="文件路径或目录路径（支持 PDF 文件）"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理，即使文件未变更（跳过完整性检查）"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="外部配置文件路径（默认使用包内置配置）"
    )

    args = parser.parse_args(argv)

    # 先检查文件路径（在加载配置和初始化 Pipeline 之前）
    path = Path(args.path)

    try:
        files_to_process = collect_pdf_files(path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if path.is_dir() and not files_to_process:
        logger.warning(f"目录中未找到 PDF 文件: {path}")
        sys.exit(0)
    if path.is_dir():
        logger.info(f"找到 {len(files_to_process)} 个 PDF 文件")

    # 加载配置
    try:
        from modular_rag_mcp_server.core.settings import load_settings
        settings = load_settings(args.config)
        logger.info(f"配置加载成功: LLM provider={settings.llm.provider}, Embedding provider={settings.embedding.provider}")
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        sys.exit(1)

    # 初始化 Pipeline
    try:
        from modular_rag_mcp_server.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline(settings)
        logger.info("Pipeline 初始化成功")
    except Exception as e:
        logger.error(f"Pipeline 初始化失败: {e}")
        sys.exit(1)

    # 处理 --force 选项
    if args.force:
        logger.info("强制模式：将跳过文件完整性检查")

    # 处理每个文件
    success_count = 0
    skip_count = 0
    error_count = 0

    # 导入 FileIntegrityChecker 用于检查文件是否会被跳过
    from modular_rag_mcp_server.ingestion.integrity import (
        FileIntegrityChecker,
        build_processing_profile,
    )
    integrity_checker = FileIntegrityChecker()
    processing_profile = build_processing_profile(
        settings,
        loader_name="pdf",
    )

    for file_path in files_to_process:
        logger.info(f"处理文件: {file_path}")

        # 检查文件是否会被跳过（除非使用 --force）
        if not args.force:
            try:
                file_hash = integrity_checker.compute_sha256(str(file_path))
                if integrity_checker.should_skip(
                    file_hash,
                    args.collection,
                    processing_profile,
                ):
                    skip_count += 1
                    logger.info(f"⏭️  文件跳过（未变更）: {file_path}")
                    continue
            except Exception as e:
                # 如果检查失败，继续处理
                logger.warning(f"完整性检查失败，继续处理: {e}")

        try:
            # 调用 Pipeline 处理
            pipeline.process(
                file_path=str(file_path),
                collection_name=args.collection,
                force=args.force,
            )
            success_count += 1
            logger.info(f"✅ 文件处理成功: {file_path}")
        except Exception as e:
            error_count += 1
            logger.error(f"❌ 文件处理失败: {file_path}, 错误: {e}")
            if args.force:
                # 在强制模式下，即使出错也继续处理其他文件
                continue
            else:
                # 非强制模式下，遇到错误可以继续或退出
                # 这里选择继续处理其他文件
                continue

    # 输出统计信息
    logger.info("=" * 60)
    logger.info("处理完成统计:")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  跳过: {skip_count}")
    logger.info(f"  失败: {error_count}")
    logger.info(f"  总计: {len(files_to_process)}")
    logger.info("=" * 60)

    # 检查输出目录（根据配置动态生成）
    if settings.vector_store.backend == "sqlite":
        sqlite_path = getattr(settings.vector_store, "sqlite_path", None)
        output_items = [Path(sqlite_path)] if sqlite_path else []
        logger.info("输出（SQLite 统一存储）:")
    else:
        output_items = [Path(settings.vector_store.persist_path)]
        logger.info("输出目录:")
    for p in output_items:
        if p.exists():
            logger.info(f"  ✅ {p}")
        else:
            logger.info(f"  ⚠️  {p} (不存在)")

    # 显式关闭资源，避免 QdrantClient 等析构时报错
    try:
        pipeline.close()
    except Exception:
        pass

    # 根据结果决定退出码
    if error_count > 0:
        sys.exit(1)
    elif success_count == 0 and skip_count == 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
