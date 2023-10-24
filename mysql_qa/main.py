# 该脚本用于: MySQL系统的入门程序.

# 导包
# 【修复】原为裸导入 `from db.mysql_client import ...`，只能在 mysql_qa/ 目录下运行。
# mysql_qa/__init__.py 已将项目根目录加入 sys.path，这里统一使用包内绝对导入，
# 从项目根目录执行 `python -m mysql_qa.main` 或 `python mysql_qa/main.py` 均可运行。
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mysql_qa.db.mysql_client import MySQLClient         # 导入 MySQL 客户端
from mysql_qa.cache.redis_client import RedisClient      # 导入 Redis 客户端
from mysql_qa.retrieval.bm25_search import BM25Search    # 导入 BM25 搜索
from base import logger     # 导入日志
import time                 # 导入时间库

