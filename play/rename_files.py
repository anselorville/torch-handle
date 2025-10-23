#!/usr/bin/env python3
"""
批量重命名文件脚本
将形如 "(100分) - VLAN资源池（Java & Python& JS & C++ & C ）.md"
改为 "VLAN资源池(100).md"
"""

import os
import re
from pathlib import Path


def extract_title_and_score(filename):
    """
    从文件名中提取题目标题和分数
    
    Args:
        filename: 原始文件名
    
    Returns:
        (title, score) 或 None
    """
    # 匹配模式：(分数) [分隔符] 题目名称（语言信息）.md
    # 支持多种分隔符：' - ', '- ', ' -', '-'
    patterns = [
        # 匹配 (100分) - 题目名称（...）.md 或 (100分)- 题目名称（...）.md
        r'^\((\d+)分\)\s*-?\s*(.+?)(?:（|[(][Jj]ava|[Pp]ython|[Jj][Ss]|[Cc]\+\+|[Cc]|&|\s).+\.md$',
        # 匹配 (100分) 题目名称 100分（...）.md
        r'^\((\d+)分\)\s*-?\s*(.+?)\s+\d+分(?:（|[(]).+\.md$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            score = match.group(1)
            title = match.group(2).strip()
            # 去除标题末尾的空格、分隔符等
            title = re.sub(r'\s+$', '', title)
            return title, score
    
    return None


def rename_files(directory):
    """
    批量重命名指定目录下的文件
    
    Args:
        directory: 目标目录路径
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"错误：目录 {directory} 不存在")
        return
    
    # 获取所有 .md 和 .html 文件（排除 README.md）
    files = [f for f in dir_path.glob('*') if f.suffix in ['.md', '.html'] and f.name != 'README.md']
    
    if not files:
        print(f"在 {directory} 中未找到需要重命名的文件")
        return
    
    print(f"找到 {len(files)} 个文件")
    print("=" * 80)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    # 用于检测重名
    new_names = {}
    
    for md_file in sorted(files):
        old_name = md_file.name
        
        # 提取标题和分数
        result = extract_title_and_score(old_name)
        
        if not result:
            print(f"⊘ 跳过: {old_name} (无法解析)")
            skip_count += 1
            continue
        
        title, score = result
        new_name = f"{title}({score}).md"
        
        # 检查是否重名
        if new_name in new_names:
            # 如果重名，添加序号
            counter = 2
            base_name = f"{title}({score})"
            while f"{base_name}_{counter}.md" in new_names:
                counter += 1
            new_name = f"{base_name}_{counter}.md"
            print(f"⚠ 检测到重名，使用: {new_name}")
        
        new_names[new_name] = True
        new_path = md_file.parent / new_name
        
        # 如果新旧文件名相同，跳过
        if old_name == new_name:
            print(f"→ 跳过: {old_name} (已是目标格式)")
            skip_count += 1
            continue
        
        try:
            md_file.rename(new_path)
            print(f"✓ {old_name}")
            print(f"  → {new_name}")
            print()
            success_count += 1
        except Exception as e:
            print(f"✗ 重命名失败: {old_name}")
            print(f"  错误: {str(e)}")
            print()
            fail_count += 1
    
    print("=" * 80)
    print(f"重命名完成！成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count}")


if __name__ == "__main__":
    # 设置目标目录
    target_directory = "/opt/workspace/git_repository/torch-handle/play/papers/2025C"
    
    print("=" * 80)
    print("文件批量重命名工具")
    print("=" * 80)
    print(f"目标目录: {target_directory}")
    print("=" * 80)
    
    rename_files(target_directory)

