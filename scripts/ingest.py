#!/usr/bin/env python3
"""
数据摄取脚本入口

支持离线批量处理文档，将文档转换为向量和索引存储。
"""
import argparse
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="数据摄取脚本：将文档转换为向量和索引存储",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件
  python scripts/ingest.py --collection my_collection --path document.pdf
  
  # 强制重新处理（忽略完整性检查）
  python scripts/ingest.py --collection my_collection --path document.pdf --force
  
  # 处理目录下的所有 PDF 文件
  python scripts/ingest.py --collection my_collection --path ./documents/
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
        default="config/settings.yaml",
        help="配置文件路径（默认: config/settings.yaml）"
    )
    
    args = parser.parse_args()
    
    # 先检查文件路径（在加载配置和初始化 Pipeline 之前）
    path = Path(args.path)
    
    if not path.exists():
        logger.error(f"路径不存在: {path}")
        sys.exit(1)
    
    # 加载配置
    try:
        from src.core.settings import load_settings
        settings = load_settings(args.config)
        logger.info(f"配置加载成功: LLM provider={settings.llm.provider}, Embedding provider={settings.embedding.provider}")
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        sys.exit(1)
    
    # 初始化 Pipeline
    try:
        from src.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline(settings)
        logger.info("Pipeline 初始化成功")
    except Exception as e:
        logger.error(f"Pipeline 初始化失败: {e}")
        sys.exit(1)
    
    # 处理 --force 选项
    if args.force:
        logger.info("强制模式：将跳过文件完整性检查")
        # TODO: 实现强制模式（可能需要修改 FileIntegrityChecker 或 Pipeline）
        # 当前实现中，Pipeline 会自动进行完整性检查
        # 如果需要强制处理，可以清除完整性检查记录或修改检查逻辑
    
    # 收集要处理的文件
    files_to_process = []
    
    if path.is_file():
        # 单个文件
        if path.suffix.lower() == ".pdf":
            files_to_process.append(path)
        else:
            logger.error(f"不支持的文件类型: {path.suffix}（当前仅支持 PDF）")
            sys.exit(1)
    elif path.is_dir():
        # 目录：查找所有 PDF 文件
        pdf_files = list(path.glob("*.pdf")) + list(path.glob("**/*.pdf"))
        if not pdf_files:
            logger.warning(f"目录中未找到 PDF 文件: {path}")
            sys.exit(0)
        files_to_process.extend(pdf_files)
        logger.info(f"找到 {len(files_to_process)} 个 PDF 文件")
    else:
        logger.error(f"无效的路径类型: {path}")
        sys.exit(1)
    
    # 处理每个文件
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 导入 FileIntegrityChecker 用于检查文件是否会被跳过
    from src.libs.loader.file_integrity import FileIntegrityChecker
    integrity_checker = FileIntegrityChecker()
    
    for file_path in files_to_process:
        logger.info(f"处理文件: {file_path}")
        
        # 检查文件是否会被跳过（除非使用 --force）
        if not args.force:
            try:
                file_hash = integrity_checker.compute_sha256(str(file_path))
                if integrity_checker.should_skip(file_hash):
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
                collection_name=args.collection
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
    
    # 检查输出目录
    output_dirs = [
        Path("data/db/chroma"),
        Path("data/db/bm25"),
        Path("data/images"),
    ]
    
    logger.info("输出目录:")
    for output_dir in output_dirs:
        if output_dir.exists():
            logger.info(f"  ✅ {output_dir}")
        else:
            logger.info(f"  ⚠️  {output_dir} (不存在)")
    
    # 根据结果决定退出码
    if error_count > 0:
        sys.exit(1)
    elif success_count == 0 and skip_count == 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
