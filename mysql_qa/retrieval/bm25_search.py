# 该脚本用于: BM25搜索, 阈值: >= 0.85才会走MySQL系统.

# 导包
import os, sys                                  # 标准库：路径与模块搜索路径
# 把"项目根目录"加入 sys.path，这样直接用 `python mysql_qa/retrieval/bm25_search.py`
# 单独运行本文件时，也能解析 `from mysql_qa.xxx import ...` 这类绝对导入。
# （原文件缺这段，导致单独运行报 ModuleNotFoundError: No module named 'mysql_qa'）
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rank_bm25 import BM25Okapi                 # 导入 BM25 算法
import numpy as np                              # 导入数值计算库
from mysql_qa.utils.preprocess import preprocess_text    # 导入文本预处理
from mysql_qa.db.mysql_client import MySQLClient         # 导入MySQL数据库操作
from mysql_qa.cache.redis_client import RedisClient      # 导入Redis缓存操作
from base import logger                         # 导入日志


# todo 1. 定义BM25搜索类 -> 封装数据加载, 模型初始化, 相似度计算, 答案检索等...
