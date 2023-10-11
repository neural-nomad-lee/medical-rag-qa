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
class BM25Search:
    # todo 1.1 初始化函数 -> 客户端实例, 模型参数, 加载数据.
    def __init__(self, redis_client, mysql_client):
        # 1. 日志实例.
        self.logger = logger
        # 2. Redis客户端实例.
        self.redis_client = redis_client
        # 3. MySQL客户端实例.
        self.mysql_client = mysql_client
        # 4. 初始化BM25模型.
        self.bm25 = None
        # 5. 初始化分词后的问题列表.
        self.questions = None
        # 6. 初始化原始问题列表.
        self.original_questions = None
        # 7. 调用数据加载方法 -> 加载问题数据, 并初始化模型.
        self._load_data()       # 小细节: 加_的函数, 表示: 内置函数

    # todo 1.2 数据加载方法 -> 优先从Reids缓存加载, 缓存未命中则从MySQL加载.
    def _load_data(self):
