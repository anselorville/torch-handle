# 2025C 题目集

本文件夹包含华为 OD 机试题目的 HTML 和 Markdown 格式文件。

## 文件说明

- **HTML 文件**：原始题目文件，包含完整的网页格式和样式
- **Markdown 文件**：转换后的纯文本格式，便于阅读和编辑

## 转换说明

所有 Markdown 文件都是通过 `convert_html_to_md.py` 脚本从 HTML 文件转换而来。

转换脚本已经进行了以下优化：
- ✅ 移除代码块中的行号列表
- ✅ 清理多余的空行
- ✅ 保留代码的原始格式和注释
- ✅ 保留标题、链接、图片等元素

## 重新转换

如果需要重新转换文件，请运行：

```bash
cd /opt/workspace/git_repository/torch-handle
python convert_html_to_md.py
```

## 题目统计

- 100分题目：25 道
- 200分题目：12 道
- 总计：37 道题目

所有题目都包含 Java、Python、JavaScript、C++、C 等多种语言的解法。

