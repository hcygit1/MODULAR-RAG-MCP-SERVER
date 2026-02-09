#!/usr/bin/env python3
"""
检查 MarkItDown 解析结果的脚本

用于查看 MarkItDown 解析 PDF 后生成的 Markdown 内容和结构。
"""
import argparse
import sys
from pathlib import Path

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    print("错误: MarkItDown 未安装。请安装: pip install markitdown")
    sys.exit(1)


def inspect_markitdown_result(pdf_path: str, output_file: str = None):
    """
    检查 MarkItDown 解析结果
    
    Args:
        pdf_path: PDF 文件路径
        output_file: 输出文件路径（可选），如果提供则保存 Markdown 到文件
    """
    if not Path(pdf_path).exists():
        print(f"错误: 文件不存在: {pdf_path}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"正在解析 PDF: {pdf_path}")
    print("=" * 80)
    
    # 初始化 MarkItDown
    md = MarkItDown()
    
    # 解析 PDF
    try:
        result = md.convert(pdf_path)
        print("\n✅ PDF 解析成功\n")
    except Exception as e:
        print(f"\n❌ PDF 解析失败: {e}\n")
        sys.exit(1)
    
    # 1. 检查 result 对象的类型和属性
    print("=" * 80)
    print("1. Result 对象信息")
    print("=" * 80)
    print(f"类型: {type(result)}")
    print(f"类型名: {type(result).__name__}")
    print(f"\n所有属性:")
    attrs = [attr for attr in dir(result) if not attr.startswith('_')]
    for attr in attrs:
        try:
            value = getattr(result, attr)
            if not callable(value):
                print(f"  - {attr}: {type(value).__name__}")
        except:
            pass
    
    # 2. 提取文本内容
    print("\n" + "=" * 80)
    print("2. Markdown 文本内容")
    print("=" * 80)
    
    # 尝试不同的方式获取文本
    markdown_text = None
    if hasattr(result, 'text_content'):
        markdown_text = result.text_content
        print("使用 result.text_content")
    elif hasattr(result, 'text'):
        markdown_text = result.text
        print("使用 result.text")
    elif hasattr(result, 'markdown'):
        markdown_text = result.markdown
        print("使用 result.markdown")
    elif isinstance(result, str):
        markdown_text = result
        print("result 是字符串类型")
    else:
        markdown_text = str(result)
        print("使用 str(result)")
    
    if markdown_text:
        print(f"\n文本长度: {len(markdown_text)} 字符")
        print(f"文本行数: {len(markdown_text.splitlines())} 行")
        print(f"\n前 500 个字符预览:")
        print("-" * 80)
        print(markdown_text[:500])
        if len(markdown_text) > 500:
            print("...")
        print("-" * 80)
        
        # 保存到文件（如果指定）
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            print(f"\n✅ Markdown 已保存到: {output_file}")
    else:
        print("⚠️  未找到文本内容")
    
    # 3. 检查图片相关信息
    print("\n" + "=" * 80)
    print("3. 图片信息检查")
    print("=" * 80)
    
    image_attrs = ['images', 'image_refs', 'image_list', 'image_paths', 'media']
    found_images = False
    
    for attr in image_attrs:
        if hasattr(result, attr):
            value = getattr(result, attr)
            print(f"\n找到属性: {attr}")
            print(f"  类型: {type(value)}")
            if isinstance(value, (list, dict)):
                print(f"  长度/大小: {len(value)}")
                if value:
                    print(f"  内容预览: {value[:3] if isinstance(value, list) else list(value.items())[:3]}")
                    found_images = True
            else:
                print(f"  值: {value}")
                if value:
                    found_images = True
    
    if not found_images:
        print("⚠️  未找到图片相关属性")
        print("   检查的属性: images, image_refs, image_list, image_paths, media")
    
    # 4. 检查其他元数据
    print("\n" + "=" * 80)
    print("4. 其他元数据")
    print("=" * 80)
    
    metadata_attrs = ['metadata', 'meta', 'info', 'properties']
    found_metadata = False
    
    for attr in metadata_attrs:
        if hasattr(result, attr):
            value = getattr(result, attr)
            print(f"\n找到属性: {attr}")
            print(f"  类型: {type(value)}")
            if isinstance(value, dict):
                print(f"  键: {list(value.keys())}")
                found_metadata = True
            else:
                print(f"  值: {value}")
                if value:
                    found_metadata = True
    
    if not found_metadata:
        print("⚠️  未找到元数据相关属性")
    
    # 5. 检查 Markdown 中的图片引用
    print("\n" + "=" * 80)
    print("5. Markdown 文本中的图片引用")
    print("=" * 80)
    
    if markdown_text:
        import re
        # 查找常见的图片引用格式
        patterns = [
            (r'!\[.*?\]\((.*?)\)', 'Markdown 图片链接'),
            (r'<img.*?src=["\'](.*?)["\'].*?>', 'HTML img 标签'),
            (r'\[IMAGE:\s*(.*?)\]', 'IMAGE 占位符'),
            (r'image[_-]?id[:\s=]+([^\s\]\)]+)', 'image_id 引用'),
        ]
        
        found_refs = False
        for pattern, desc in patterns:
            matches = re.findall(pattern, markdown_text, re.IGNORECASE)
            if matches:
                print(f"\n找到 {desc}:")
                for match in matches[:5]:  # 只显示前 5 个
                    print(f"  - {match}")
                if len(matches) > 5:
                    print(f"  ... 还有 {len(matches) - 5} 个")
                found_refs = True
        
        if not found_refs:
            print("⚠️  未在 Markdown 文本中找到图片引用")
    
    # 6. 完整对象转储（用于调试）
    print("\n" + "=" * 80)
    print("6. Result 对象完整信息（用于调试）")
    print("=" * 80)
    print(f"类型: {type(result)}")
    print(f"字符串表示: {str(result)[:200]}...")
    
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="检查 MarkItDown 解析 PDF 的结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python scripts/inspect_markitdown.py path/to/file.pdf
  
  # 保存 Markdown 到文件
  python scripts/inspect_markitdown.py path/to/file.pdf --output output.md
        """
    )
    
    parser.add_argument(
        "pdf_path",
        type=str,
        help="PDF 文件路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出 Markdown 文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    inspect_markitdown_result(args.pdf_path, args.output)


if __name__ == "__main__":
    main()
