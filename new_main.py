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
        # 3. 初始化大语言模型 -> LLM客户端.
        try:
            # 初始化 OpenAI 客户端，连接 DashScope API
            self.client = OpenAI(api_key=self.config.DASHSCOPE_API_KEY, base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            # 记录 OpenAI 初始化失败的错误日志
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            # 抛出异常，终止初始化
            raise

        # 4. 初始化向量数据库(用于 RAG 系统的知识库管理) 和RAG系统
        self.vector_store = VectorStore()  # 创建向量数据库实例 -> 包括: 向量存储 和 检索.
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)  # 创建 RAG 系统实例 -> 包括: 检索和生成.

        # 5. (优化1) 初始化对话历史表: 在MySQL中创建存储会话记录的表.
        self.init_conversation_table()

    # todo 2.2 (优化2) 初始化对话历史表的方法: 在MySQL中创建存储对话记录的表结构.
    def init_conversation_table(self):
        """
        函数功能: 初始化MySQL中的 Conversations表, 用于存储对话历史.
        :return:
        """
        try:
            # 1. 执行SQL语句, 创建 conversations 表，包含会话 ID、问题、答案和时间戳
            self.mysql_client.cursor.execute("""
                create table if not exists conversations(
                    id INT AUTO_INCREMENT PRIMARY KEY,          # 主键id
                    session_id VARCHAR(36) NOT NULL,            # 会话id
                    question TEXT NOT NULL,                     # 问题
                    answer TEXT NOT NULL,                       # 答案
                    timestamp DATETIME NOT NULL,                # 时间戳
                    INDEX idx_session_id (session_id)           # 创建索引列, 目的: 提高查询的效率.
                )
            """)
            # 提交数据库事务
            self.mysql_client.connection.commit()
            # 记录表初始化成功的日志
            self.logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            # 记录表初始化失败的错误日志
            self.logger.error(f"初始化对话历史表失败: {e}")
