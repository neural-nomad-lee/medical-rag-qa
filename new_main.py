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
            # 抛出异常，终止初始化
            raise

    # todo 2.2.1 (新功能) 知识沉淀: RAG检索超过3次的问题, 结构化后存入MySQL知识库.
    # 设计说明:
    #   · 用 Redis 计数器(30天滑动窗口)统计同一问题的RAG检索次数.
    #   · 超过3次说明该问题高频且有真实价值 -> 将"问题-答案"结构化写入 ylkb 表,
    #     之后相同问题可直接命中 BM25/MySQL 精准问答链路, 不再重复走RAG生成.
    #   · 写入的答案设置 expire_at = 当前时间 + 180天(半年), 到期由 cleanup_expired_knowledge 自动清理.
    #   · 同一问题若已有旧答案 -> 先删除再插入(覆盖旧资料, 保证答案为最新).
    RAG_PERSIST_THRESHOLD = 3          # 沉淀阈值: 检索次数 > 3 次才存入MySQL
    RAG_PERSIST_WINDOW_SECONDS = 30 * 24 * 3600   # 计数器窗口: 30天
    KNOWLEDGE_EXPIRE_DAYS = 180        # 沉淀知识的到期天数: 半年

    def persist_rag_answer(self, query: str, answer: str, department=None) -> bool:
        """
        函数功能: 将高频RAG问答结构化存入MySQL知识库(ylkb表), 半年后到期自动清理.
        :param query: 用户问题原文
        :param answer: RAG生成的完整答案
        :param department: 科室过滤条件(可为None, 存"通用")
        :return: True -> 写入成功(或无需写入), False -> 写入失败
        """
        # 1. 无有效答案则不沉淀.
        if not answer or not answer.strip():
            return True
        try:
            # 2. Redis计数: 统计该问题的RAG检索次数(每次RAG成功生成答案+1).
            #   Redis 属于辅助组件, 计数失败只跳过沉淀, 不能影响已经生成的答案.
            count_key = f"rag_query_count:{query}"
            try:
                count = int(self.redis_client.client.incr(count_key))
                self.redis_client.client.expire(count_key, self.RAG_PERSIST_WINDOW_SECONDS)
            except redis.RedisError as e:
                self.logger.warning(f"RAG知识沉淀计数失败(跳过本次沉淀): {e}")
                return True
            # 3. 未超过阈值 -> 只计数, 不写入.
            if count <= self.RAG_PERSIST_THRESHOLD:
                self.logger.info(f"RAG知识沉淀计数: '{query}' 已检索 {count} 次 (阈值 {self.RAG_PERSIST_THRESHOLD})")
                return True
            # 4. 超过阈值 -> 结构化写入MySQL: 同问题旧答案直接覆盖.
            dept = department or "通用"
            self.mysql_client.cursor.execute(
                "DELETE FROM ylkb WHERE question = %s", (query,)
            )
            self.mysql_client.cursor.execute(
                "INSERT INTO ylkb(department, question, answer, expire_at) "
                "VALUES(%s, %s, %s, DATE_ADD(NOW(), INTERVAL %s DAY))",
                (dept, query, answer, self.KNOWLEDGE_EXPIRE_DAYS)
            )
            self.mysql_client.connection.commit()
            self.logger.info(f"RAG知识已沉淀入MySQL: '{query}' (科室: {dept}, {self.KNOWLEDGE_EXPIRE_DAYS}天后到期)")
            # 5. 刷新BM25缓存, 让沉淀的知识立即可被精准问答链路命中.
            self.bm25_search.refresh()
            return True
        except pymysql.MySQLError as e:
            self.logger.error(f"RAG知识沉淀失败: {e}")
            try:
                self.mysql_client.connection.rollback()
            except Exception:
                pass
            return False

