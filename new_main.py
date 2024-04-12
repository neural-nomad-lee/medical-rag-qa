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

    # todo 2.2.2 (新功能) 到期知识清理: 删除MySQL中已到期(半年)的沉淀知识, 并失效BM25缓存.
    def cleanup_expired_knowledge(self) -> int:
        """
        函数功能: 删除 ylkb 表中 expire_at 已到期的记录(RAG沉淀知识半年到期), 返回删除条数.
        说明: FAQ种子数据 expire_at 为 NULL, 永久保留, 不受本清理影响.
        :return: 删除的记录条数(出错时返回-1)
        """
        try:
            self.mysql_client.cursor.execute(
                "DELETE FROM ylkb WHERE expire_at IS NOT NULL AND expire_at < NOW()"
            )
            deleted = self.mysql_client.cursor.rowcount
            self.mysql_client.connection.commit()
            if deleted > 0:
                self.logger.info(f"已清理 {deleted} 条到期知识")
                # 有删除才需要刷新BM25缓存(问题列表可能变化).
                self.bm25_search.refresh()
            return deleted
        except pymysql.MySQLError as e:
            self.logger.error(f"清理到期知识失败: {e}")
            try:
                self.mysql_client.connection.rollback()
            except Exception:
                pass
            return -1

    # todo 2.3 (优化3) 调用大语言模型的方法: 给RAG系统用的.
    def call_dashscope(self, prompt):
        """
        函数功能: 调用DashScope API生成答案 -> 流式输出
        :param prompt: 发送给大模型的提示文本(字符串), 包含用户问题及上下文
        :return: 生成器, 逐段返回大模型生成的答案内容(字符串)
        """
        try:
            # 1. 发送聊天完成请求，启用流式输出, 设置超时时间.
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,  # 使用配置中的语言模型
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},  # 系统提示
                    {"role": "user", "content": prompt},  # 用户输入的提示
                ],
                timeout=30,  # 设置 30 秒超时
                stream=True  # 启用流式输出
            )
            # 2. 遍历流式输出的每个 chunk
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    # 获取当前 chunk 的文本内容
                    content = chunk.choices[0].delta.content
                    # 逐 token 返回，供前端实时显示
                    yield content
        except Exception as e:
            # 记录 API 调用失败的错误日志
            self.logger.error(f"LLM调用失败: {e}")
            # 返回错误信息
            return f"错误：LLM调用失败 - {e}"


    # todo 2.4 (优化4) 获取最近对话历史的内部方法: 从MySQL中查询指定会话(ID)的最近5轮对话.
    def _fetch_recent_history(self, session_id):
        """
        函数功能: 获取最近5轮对话历史
        :param session_id: 会话唯一标识(字符串), 用于定位具体会话.
        :return: 列表, 包含字典形式的对话记录, 例如: [{'question':问题, 'answer':答案}]
        """
        try:
            # 1. 执行 SQL 查询，获取最近 5 轮对话
            self.mysql_client.cursor.execute("""
                 SELECT question, answer
                 FROM conversations
                 WHERE session_id = %s
                 ORDER BY timestamp DESC
                     LIMIT %s
                 """, (session_id, 5))
            # 2. 将查询结果转换为字典列表
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            # 3. 反转结果，按时间正序返回
            return history[::-1]
        except pymysql.MySQLError as e:
            # 记录查询失败的错误日志
            self.logger.error(f"获取对话历史失败: {e}")
            # 返回空列表
            return []


    # todo 2.5 (优化5) 获取会话历史的方法: 对外提供获取指定会话历史的接口.
    def get_session_history(self, session_id):
        """
        函数功能: 从MySQL获取会话历史.
        :param session_id: 会话唯一标识(字符串)
        :return: 列表, 包含最近5轮对话记录 -> 同 _fetch_recent_history()的返回值.
        """
        # 调用内部方法获取对话历史.
        return self._fetch_recent_history(session_id)


    # todo 2.6 (优化6) 更新会话历史的方法: 将新对话记录存入MySQL, 并保留最近5轮.
    def update_session_history(self, session_id: str, question: str, answer: str):
        """
        函数功能: 更新会话历史到MySQL，保留最近5轮对话
        :param session_id: 会话唯一标识(字符串)
        :param question: 用户的问题(字符串)
        :param answer: 系统生成的答案(字符串)
        :return:
        """
        try:
            # 1. 插入新的对话记录, 将问题, 答案及当前时间存入数据库.
            self.mysql_client.cursor.execute("""
                 INSERT INTO conversations (session_id, question, answer, timestamp)
                 VALUES (%s, %s, %s, NOW())
                 """, (session_id, question, answer))
            # 2. 获取更新后的对话历史
            history = self._fetch_recent_history(session_id)
            # 3. 删除超出 5 轮的旧记录
            self.mysql_client.cursor.execute("""
                 DELETE
                 FROM conversations
                 WHERE session_id = %s
                   AND id NOT IN (SELECT id
                      FROM (SELECT id
                            FROM conversations
                            WHERE session_id = %s
                            ORDER BY timestamp DESC
                                LIMIT %s) AS sub)
                 """, (session_id, session_id, 5))
            # 4. 提交事务
            self.mysql_client.connection.commit()
            # 5. 记录更新成功的日志
            self.logger.info(f"会话 {session_id} 历史更新成功")
            # 6. 返回更新后的历史
            return history
        except pymysql.MySQLError as e:
            # 7.记录数据库操作失败的错误日志
            self.logger.error(f"更新会话历史失败: {e}")
            # 回滚事务
            self.mysql_client.connection.rollback()
            # 抛出异常
            raise
        except Exception as e:
            # 8.记录意外错误的日志
            self.logger.error(f"更新会话历史意外错误: {e}")
            # 回滚事务
            self.mysql_client.connection.rollback()
            # 抛出异常
            raise


    # todo 2.7 (优化7) 清除会话历史: 删除指定会话的所有记录.
    def clear_session_history(self, session_id: str) -> bool:
        """
        函数功能: 清除指定会话历史
        :param session_id: 会话唯一标识(字符串)
        :return: True -> 清除成功, False -> 清除失败
        """
        try:
            # 1.删除指定 session_id 的所有对话记录
            self.mysql_client.cursor.execute("""
                 DELETE
                 FROM conversations
                 WHERE session_id = %s
                 """, (session_id,)
            )
            # 2.提交事务
            self.mysql_client.connection.commit()
            # 3.记录清除成功的日志
            self.logger.info(f"会话 {session_id} 历史已清除")
            # 4.返回 True 表示成功
            return True
        except pymysql.MySQLError as e:
            # 5.记录清除失败的错误日志
            self.logger.error(f"清除会话历史失败: {e}")
            # 回滚事务
            self.mysql_client.connection.rollback()
            # 返回 False 表示失败
            return False


    # todo 2.8 (优化8) 处理查询的核心方法: 整合检索, 大模型生成和对话历史, 返回流式答案.
    def query(self, query, source_filter=None, session_id=None):
        """
        函数功能: 查询集成系统，支持对话历史和流式输出
        :param query:
        :param source_filter:
        :param session_id:
        :return:
        """
        # 1.记录查询开始时间 和 日志.
        start_time = time.time()
        # 记录查询信息到日志
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")

        # 2.获取对话历史，若无 session_id 则返回空列表, 若有则查询最近5轮历史.
        history = self.get_session_history(session_id) if session_id else []
        # 3. 执行 BM25 搜索，获取答案和是否需要 RAG 的标志
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)

        if answer:
            # 情况1: 找到可靠答案 -> 即: BM25检索结果满足阈值. 记录答案到日志
            self.logger.info(f"MySQL答案: {answer}")
            if session_id:
                # 更新对话历史
                self.update_session_history(session_id, query, answer)
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 一次性返回答案，标记为完整
            yield answer, True
        elif need_rag:
            # 情况2: 无可靠答案 -> 回退到RAG系统生成.
            self.logger.info("无可靠MySQL答案，回退到RAG")
            # 初始化收集完整答案的字符串
            collected_answer = ""
            # 从 RAG 系统获取流式输出
            for token in self.rag_system.generate_answer(query, source_filter=source_filter, history=history):
                # 累积答案
                collected_answer += token
                # 逐 token 返回，标记为部分答案
                yield token, False
            if session_id:
                # 更新对话历史，存储完整答案
                self.update_session_history(session_id, query, collected_answer)
            # (新功能)知识沉淀: 高频RAG问题结构化写入MySQL知识库, 半年后到期自动清理.
            self.persist_rag_answer(query, collected_answer, source_filter)
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 返回空字符串，标记流结束
            yield "", True
        else:
            # 情况3: 未找到任何答案.
            self.logger.info("未找到答案")
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 一次性返回默认答案，标记为完整
            yield "未找到答案", True


# todo 3. 定义main函数: 提供命令行交互界面, 测试 集成问答系统.
def main():
    # 1. 定义主函数，提供命令行交互界面 -> 初始化问答系统.
    qa_system = IntegratedQASystem()  # 初始化问答系统
    # 2. 生成唯一会话 ID
    session_id = str(uuid.uuid4())
    # 3. 打印欢迎信息
    print("\n欢迎使用 MedRAG 医疗智慧问答系统！")
    # 打印会话 ID
    print(f"会话ID: {session_id}")
    # 打印支持的科室类别
    print(f"支持的科室类别：{qa_system.config.DEPARTMENTS}")
    # 提示用户输入查询或退出
    print("输入查询进行问答，输入 'exit' 退出。")

    try:
        while True:
            # 4. 获取用户输入的查询
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                # 如果用户输入 exit，记录退出日志
                logger.info("退出系统")
                # 打印退出信息
                print("再见, 感谢您的使用, 期待下次再会！")
                # 退出循环
                break
            # 5. 获取用户输入的科室过滤 -> 允许用户指定科室类别, 无效则忽略.
            # 5.1 接收用户录入的科室类别
            source_filter = input(f"请输入科室类别 ({'/'.join(qa_system.config.DEPARTMENTS)}) (直接回车默认不过滤): ").strip()
            # 5.2 判断用户录入的 科室类别 是否有效.
            if source_filter and source_filter not in qa_system.config.DEPARTMENTS:
                # 如果科室过滤无效，记录警告日志
                logger.warning(f"无效的科室类别 '{source_filter}'，将不过滤")
                # 设置为空，忽略过滤
                source_filter = None
            # 6. 打印答案提示
            print("\n答案: ", end="", flush=True)
            # 初始化累积答案的字符串
            answer = ""
            # 迭代 query 方法的生成器
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:
                    # 仅当 token 非空时打印
                    print(token, end="", flush=True)
                    # 累积答案
                    answer += token
                if is_complete:
                    # 如果是完整答案或流结束，换行并退出循环
                    print()
                    break
            # 7. 打印对话历史
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                # 按顺序打印历史记录
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")
    except Exception as e:
        # 记录系统错误日志
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 关闭 MySQL 连接
        qa_system.mysql_client.close()


# todo 4. 程序的主入口, 当脚本执行运行时, 执行main函数.
if __name__ == '__main__':
    main()

