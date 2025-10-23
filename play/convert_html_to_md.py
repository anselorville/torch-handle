#!/usr/bin/env python3
"""
HTML 转 Markdown 批量转换脚本
将指定文件夹下的所有 HTML 文件转换为 Markdown 格式
优化版：清理代码块行号和不必要的内容
"""

import os
import re
from pathlib import Path

try:
    import html2text
    from bs4 import BeautifulSoup
except ImportError:
    print("未安装必要的库，正在尝试安装...")
    import subprocess
    subprocess.check_call(["pip", "install", "html2text", "beautifulsoup4", "lxml"])
    import html2text
    from bs4 import BeautifulSoup


def clean_html_content(html_content):
    """
    清理 HTML 内容，去除行号等不必要的元素
    
    Args:
        html_content: 原始 HTML 内容
    
    Returns:
        清理后的 HTML 内容
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    # 移除代码行号相关的元素
    # 通常行号在 <ol> 或特定 class 的元素中
    for line_numbers in soup.find_all(['ol', 'ul'], class_=lambda x: x and ('line-numbers' in x or 'linenums' in x)):
        line_numbers.decompose()
    
    # 移除包含纯数字列表项的父元素（通常是行号）
    for ul in soup.find_all('ul'):
        items = ul.find_all('li', recursive=False)
        if len(items) > 10:  # 如果列表项超过10个
            # 检查是否所有项都是纯数字
            all_numbers = True
            for item in items[:20]:  # 检查前20个
                text = item.get_text(strip=True)
                if not text.isdigit():
                    all_numbers = False
                    break
            if all_numbers:
                ul.decompose()
    
    # 移除 <!-- --> 注释内容
    for comment in soup.find_all(text=lambda text: isinstance(text, str) and '<!--' in text):
        comment.extract()
    
    # 移除空的 span 标签
    for span in soup.find_all('span'):
        if not span.get_text(strip=True):
            span.unwrap()
    
    return str(soup)


def clean_markdown_content(markdown_content):
    """
    清理 Markdown 内容
    
    Args:
        markdown_content: 原始 Markdown 内容
    
    Returns:
        清理后的 Markdown 内容
    """
    lines = markdown_content.split('\n')
    cleaned_lines = []
    in_number_list = False
    number_list_count = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 检测是否是纯数字列表项的开始
        if re.match(r'^\s*\*\s+\d+\s*$', line):
            if not in_number_list:
                in_number_list = True
                number_list_count = 1
            else:
                number_list_count += 1
            
            # 如果连续的数字列表项超过5个，则认为是行号，跳过
            if number_list_count > 5:
                i += 1
                continue
        else:
            # 如果之前在数字列表中，且累计超过5项，则删除这些项
            if in_number_list and number_list_count > 5:
                # 删除之前添加的数字列表项
                cleaned_lines = cleaned_lines[:-min(number_list_count-1, len(cleaned_lines))]
            in_number_list = False
            number_list_count = 0
        
        # 移除多余的空行（连续超过2个空行）
        if not stripped:
            if len(cleaned_lines) > 0 and not cleaned_lines[-1].strip():
                if len(cleaned_lines) > 1 and not cleaned_lines[-2].strip():
                    i += 1
                    continue
        
        cleaned_lines.append(line)
        i += 1
    
    # 最后检查，如果末尾有数字列表，也删除
    if in_number_list and number_list_count > 5:
        cleaned_lines = cleaned_lines[:-min(number_list_count, len(cleaned_lines))]
    
    return '\n'.join(cleaned_lines)


def convert_html_to_markdown(html_file_path, output_dir=None):
    """
    将单个 HTML 文件转换为 Markdown
    
    Args:
        html_file_path: HTML 文件路径
        output_dir: 输出目录，如果为 None 则输出到与 HTML 文件相同的目录
    
    Returns:
        转换后的 Markdown 文件路径
    """
    # 读取 HTML 文件
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 清理 HTML 内容
    html_content = clean_html_content(html_content)
    
    # 创建 html2text 转换器
    h = html2text.HTML2Text()
    
    # 配置转换选项
    h.ignore_links = False  # 保留链接
    h.ignore_images = False  # 保留图片
    h.ignore_emphasis = False  # 保留强调标记
    h.body_width = 0  # 不自动换行
    h.unicode_snob = True  # 使用 Unicode
    h.skip_internal_links = False  # 不跳过内部链接
    
    # 转换为 Markdown
    markdown_content = h.handle(html_content)
    
    # 清理 Markdown 内容
    markdown_content = clean_markdown_content(markdown_content)
    
    # 确定输出路径
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        md_file_path = output_path / (Path(html_file_path).stem + '.md')
    else:
        md_file_path = Path(html_file_path).with_suffix('.md')
    
    # 写入 Markdown 文件
    with open(md_file_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return md_file_path


def batch_convert_html_to_markdown(input_dir, output_dir=None):
    """
    批量转换指定目录下的所有 HTML 文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为 None 则输出到输入目录
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"错误：目录 {input_dir} 不存在")
        return
    
    # 查找所有 HTML 文件
    html_files = list(input_path.glob('*.html'))
    
    if not html_files:
        print(f"在 {input_dir} 中未找到 HTML 文件")
        return
    
    print(f"找到 {len(html_files)} 个 HTML 文件")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    for html_file in html_files:
        try:
            md_file = convert_html_to_markdown(html_file, output_dir)
            print(f"✓ 转换成功: {html_file.name} -> {md_file.name}")
            success_count += 1
        except Exception as e:
            print(f"✗ 转换失败: {html_file.name} - 错误: {str(e)}")
            fail_count += 1
    
    print("-" * 60)
    print(f"转换完成！成功: {success_count}, 失败: {fail_count}")


if __name__ == "__main__":
    # 设置输入目录
    input_directory = "/opt/workspace/git_repository/torch-handle/play/papers/2025C"
    
    # 可选：设置输出目录（如果为 None，则在原目录生成）
    output_directory = None
    
    print("=" * 60)
    print("HTML 转 Markdown 批量转换工具")
    print("=" * 60)
    print(f"输入目录: {input_directory}")
    print(f"输出目录: {output_directory if output_directory else '与输入目录相同'}")
    print("=" * 60)
    
    batch_convert_html_to_markdown(input_directory, output_directory)

