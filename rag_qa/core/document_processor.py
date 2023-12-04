# 该脚本用于: 处理输入文档, 分块并准备向量存储.

# 导包
import os       # 文件操作

# ============================================================
# 【环境兜底】让 UnstructuredMarkdownLoader 能找到 NLTK 数据
# ------------------------------------------------------------
# 现象：加载 .md 文件时抛 LookupError: Resource punkt_tab not found
# 原因：langchain 的 UnstructuredMarkdownLoader 内部依赖 NLTK 分词/词性标注数据，
#       而 NLTK 默认只在 ~/nltk_data 等系统路径查找。
# 影响：.md 文件会被 except 分支静默跳过，导致"科室导诊指南"这类关键资料没进向量库，
#       而日志里只留一行 error，表现为"RAG 答不出大纲内容"却查不出原因。
# 解法：把项目内的 nltk_data 目录显式加入 NLTK 搜索路径（数据随项目走，换机器无需重下）。
# ============================================================
# 注意：本文件位于 rag_qa/core/，要 dirname 三次才到项目根：
#   __file__ → .../rag_qa/core/document_processor.py
#   第 1 次 → .../rag_qa/core  ｜ 第 2 次 → .../rag_qa ｜ 第 3 次 → .../integrated_qa_system（项目根）
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
