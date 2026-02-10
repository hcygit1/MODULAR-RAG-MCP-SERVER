#!/usr/bin/env python3
"""
测试 PDF 图片提取功能

用于验证 PdfLoader 的图片提取和占位符插入：
- 加载测试 PDF
- 验证图片提取
- 验证占位符插入
- 输出统计信息
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.libs.loader.pdf_loader import PdfLoader, PYMUPDF_AVAILABLE, MARKITDOWN_AVAILABLE
from src.ingestion.models import Document


def run_test(pdf_path: str) -> None:
    """
    运行图片提取测试
    
    Args:
        pdf_path: PDF 文件路径
    
    Raises:
        SystemExit: 当测试失败时
    """
    if not Path(pdf_path).exists():
        print(f"❌ 错误: 文件不存在: {pdf_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("PDF 图片提取测试")
    print("=" * 60)
    print(f"文件: {pdf_path}\n")
    
    if not MARKITDOWN_AVAILABLE:
        print("❌ MarkItDown 未安装，无法运行测试")
        sys.exit(1)
    
    if not PYMUPDF_AVAILABLE:
        print("⚠️  PyMuPDF 未安装，图片提取将使用回退策略（无图片）")
    else:
        print("✅ PyMuPDF 已安装")
    
    try:
        loader = PdfLoader()
        document = loader.load(pdf_path)
        
        print("\n--- 解析结果 ---")
        print(f"文档 ID: {document.id}")
        print(f"文本长度: {len(document.text)} 字符")
        
        # 验证图片数据
        image_data = document.metadata.get("image_data", {})
        images = document.metadata.get("images", [])
        
        print(f"\n图片数量: {len(image_data)} 张")
        print(f"图片元数据条数: {len(images)} 条")
        
        if image_data:
            total_bytes = sum(len(b) for b in image_data.values())
            print(f"图片总大小: {total_bytes} 字节")
            for img_id, data in list(image_data.items())[:5]:
                print(f"  - {img_id}: {len(data)} 字节")
            if len(image_data) > 5:
                print(f"  ... 及其他 {len(image_data) - 5} 张")
        
        # 验证占位符
        import re
        placeholders = re.findall(r'\[IMAGE:\s*([^\]]+)\]', document.text)
        print(f"\n占位符数量: {len(placeholders)} 个")
        if placeholders:
            for ph in placeholders[:5]:
                print(f"  - [IMAGE: {ph}]")
            if len(placeholders) > 5:
                print(f"  ... 及其他 {len(placeholders) - 5} 个")
        
        print("\n✅ 测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="测试 PDF 图片提取功能")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="tests/fixtures/sample_documents/sample.pdf",
        help="PDF 文件路径（默认: tests/fixtures/sample_documents/sample.pdf）"
    )
    args = parser.parse_args()
    
    run_test(args.pdf_path)


if __name__ == "__main__":
    main()
