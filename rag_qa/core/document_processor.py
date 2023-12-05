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
)
_NLTK_DIR = os.path.join(_PROJECT_ROOT, "nltk_data")
if os.path.isdir(_NLTK_DIR):
    os.environ.setdefault("NLTK_DATA", _NLTK_DIR)   # 仅在用户未设置时生效，不覆盖用户配置
    try:
        import nltk
        if _NLTK_DIR not in nltk.data.path:
            nltk.data.path.insert(0, _NLTK_DIR)      # 双保险：直接注入 nltk 搜索路径
    except ImportError:
        pass

from langchain_community.document_loaders import TextLoader     # 加载纯文本文件(.txt)用的工具
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader    # 加载(.md)文件的工具, 保留标题,列表等结构化信息
from langchain_text_splitters import MarkdownTextSplitter       # 针对 .md文档的文本分割器.
from datetime import datetime               # 用于记录文档加载/处理时间.
from rag_qa.med_text_spliter import ChineseRecursiveTextSplitter        # 适配中文医疗文本层级结构
from rag_qa.med_text_spliter import AliTextSplitter                     # 阿里中文优化(医疗场景专用文本分割器)
from rag_qa.med_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader  # 医疗场景专用文档加载器(支持OCR, 处理扫描版病历、检查报告、医学影像文字等.)
from base.config import Config      # 配置类
