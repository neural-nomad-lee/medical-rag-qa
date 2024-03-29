# 该脚本是基于 old_main.py做的优化, 在其基础上添加了: 历史对话 + 流式输出.
# 你可以把这个版本理解为: 融合了MySQL的RAG系统 V2版本
# V2版本只需要优化3个地方:
#   1. 在 integrated_qa_system项目目录下创建 new_main.py 文件.
#   2. 优化 rag_qa/core/rag_system.py的代码, 建议拷贝一份然后改名叫 new_rag_system.py, 然后在其中修改即可.
#   3. 优化 rag_qa/__init__.py的代码, 导包从rag_system 改为 new_rag_system.

import warnings
warnings.filterwarnings("ignore")

# todo 1. 导入所需的库.
# 导入 MySQL 系统组件，用于数据库操作和搜索
from mysql_qa import MySQLClient, RedisClient, BM25Search
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa import VectorStore, RAGSystem

# 导入配置和日志工具，用于系统配置和日志记录
from base.config import config
from base.logger import logger
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI
# 导入时间库，用于记录处理时间
import time
import pymysql      # 这个库是Python操作MySQL的库, 用于数据库操作的异常捕获, 增删改查操作等.
import redis        # 用于捕获 Redis 操作异常(缓存/计数失败不应影响问答主流程).
import uuid         # 生成唯一会话ID, 标识: 用户会话.


# todo 2. 集成问答系统的核心类.
# 该类的作用: 集成问答系统的核心功能(主类), 把各种工具拼起来干活.
# 先查MySQL中的现成答案 -> 靠谱就用 -> 不靠谱就用RAG重新生成 -> 啥都没有就说 没找到.
class IntegratedQASystem:
    # todo 2.1 初始化方法 -> 创建系统需要的各种工具.
    def __init__(self):
        # 1. 初始化日志工具，用于记录系统运行信息 和 配置对象.
        self.logger = logger
        self.config = config
        # 2. 初始化 MySQL 客户端 和 Redis客户端(缓存), BM25 搜索模块
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
