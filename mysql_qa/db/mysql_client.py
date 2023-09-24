# 该脚本用于: MySQL数据库的基础操作.

# 导入MySQL操作和数据处理依赖库.
import pymysql              # 用于构建MySQL数据库连接
import pandas as pd         # 用于数据处理
import sys, os              # sys用于管理系统路径, os用于处理文件刘静.

# todo 1.路径配置.
# 1. 获取当前文件所在目录的绝对路径, 即: 当前脚本所在的文件夹.
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir: {current_dir}')    # D:\workspace\ai_30_bj\integrated_qa_system\mysql_qa\db

# 2.获取当前目录的上一级目录.
module_dir = os.path.dirname(current_dir)
# print(f'module_dir: {module_dir}')      # D:\workspace\ai_30_bj\integrated_qa_system\mysql_qa

# 3. 获取项目的根目录.
project_root = os.path.dirname(module_dir)
# print(f'project_root: {project_root}')  # D:\workspace\ai_30_bj\integrated_qa_system

# 4. 将项目根目录加入系统路径, 确保跨目录导入自定义模块(例如: config, logger)...
sys.path.insert(0, project_root)

# 5. 导入自定义模块: 配置文件管理模块.
from base import Config, logger


# todo 2. 定义MySQL客户端类 -> 封装数据库连接, 表操作, 数据增查, 连接关闭等功能.
class MySQLClient:
    # todo 2.1 初始化方法 -> 建立数据库连接, 创建游标, 处理连接异常.
    def __init__(self):
        # 1. 将全局logger赋值为类属性.
        self.logger = logger
        try:
            # 2. 建立MySQL数据库连接 -> 从config中读取编码.
            # 注意: 必须显式指定 charset='utf8mb4', 本机 MySQL 5.7 服务端默认 latin1,
            #       不指定会导致中文写入报 Incorrect string value 或乱码.
            self.connection = pymysql.connect(
                host = Config().MYSQL_HOST,             # MySQL服务器主机地址
                user = Config().MYSQL_USER,             # MySQL登录用户名
                password = Config().MYSQL_PASSWORD,     # MySQL登录密码
                database = Config().MYSQL_DATABASE,     # 数据库名称
                charset = 'utf8mb4'                     # 字符集: 支持中文与emoji
            )
            # 3. 创建游标对象 -> 用于执行SQL语句并返回结果.
            self.cursor = self.connection.cursor()

            # 4. 记录info日志 -> 确认MySQL连接成功.
            self.logger.info('MySQL 连接成功')
        except pymysql.MySQLError as e:
            # 5. 记录error日志 -> 输出错误信息.
            self.logger.error(f'MySQL 连接失败: {e}')
            raise       # 抛出异常, 不屏蔽错误, 让调用方感知连接失败.


    # todo 2.2 创建数据表 -> 若 ylkb表不存在则创建, 定义表结构.
    def create_table(self):
        # 1. 定义创建表的SQL语句, 创建 ylkb表, 存储: 医疗导诊知识问答数据.
        # 注意: MySQL 5.7 不支持列定义后跟 # 注释的写法(会抛1064), 注释全部外移.
        #       answer 使用 TEXT 而非 VARCHAR(1000), 避免长答案被截断.
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ylkb (
            id INT AUTO_INCREMENT PRIMARY KEY,
            department VARCHAR(20),
            question TEXT,
            answer TEXT,
            expire_at DATETIME NULL)
        """
        # 字段说明:
        #   department 科室名称(对应医疗导诊FAQ.csv 的"科室"列, VARCHAR(20) 足够)
        #   question   问题原文
        #   answer     答案原文(用 TEXT 避免截断)
        #   expire_at  到期时间(NULL=永久保留; 由RAG沉淀的答案半年后到期自动清理)

        try:
             # 2. 执行SQL语句 -> 创建表.
             self.cursor.execute(create_table_query)
             # 3. 提交事务 -> 确认表创建成功.  MySQL默认事务需要手动提交, 确保表创建成功.
             self.connection.commit()
             # 4. 输出信息日志 -> 确认表创建成功.
             self.logger.info('表创建成功')
        except pymysql.MySQLError as e:
            # 5. 输出错误信息.
            self.logger.error(f'创建表失败: {e}')
            raise


    # todo 2.3 从csv文件批量插入数据到 ylkb表中.
    def insert_data(self, csv_path):
        try:
            # 1. 读取csv文件 -> 获取数据.
            data = pd.read_csv(csv_path)
            # 扩展: 打印下, 查看一下数据.
            # print(data.head())

            # 2. 循环遍历csv每一行数据 -> 执行插入操作.
            for _, row in data.iterrows():
                # 2.1 定义插入SQL语句, 使用 %s占位符, 避免SQL注入攻击问题.
                insert_query = "insert into ylkb(department, question, answer) values(%s, %s, %s)"
                # 2.2 执行SQL语句 -> 将csv行数据(科室, 问题, 答案)传入占位符.
                self.cursor.execute(insert_query, (row['科室'], row['问题'], row['答案']))

            # 3. 提交事务: 确保所有插入操作批量生效.
            self.connection.commit()
            # 4. 输出信息日志 -> 确认数据插入成功.
            self.logger.info('MySQL数据插入成功')

        except Exception as e:
            # 5.捕获异常信息, 回滚事务, 抛出异常信息.
            # 5.1 输出错误信息.
            self.logger.error(f'数据插入失败: {e}')
            # 5.2 事务回滚: 把数据库连接恢复到事务开始前的状态 -> 类似于: Linux的快照.
            self.connection.rollback()
            # 5.3 抛出异常, 不屏蔽错误, 让调用方感知数据插入失败.
            raise


    # todo 2.4 从 ylkb表中获取所有问题(返回问题列表)
    def fetch_questions(self):
        try:
            # 1. 执行SQL查询.
            self.cursor.execute("select question from ylkb")
            # 2. 获取查询结果. fetch_all(): 返回所有匹配数据.
            results = self.cursor.fetchall()
            # 3. 记录INFO日志 -> 确认数据查询成功.
            self.logger.info('数据查询成功')
            # 4. 返回查询结果.
            return results

        except pymysql.MySQLError as e:
            # 5. 输出错误信息.
            self.logger.error(f'数据查询失败: {e}')
            # 返回空列表, 避免异常.
            return []


    # todo 2.5 根据指定问题 从 ylkb表中获取对应的答案.
    def fetch_answer(self, question):
        try:
            # 1. 执行SQL查询.
            self.cursor.execute("select answer from ylkb where question=%s", (question,))       # 占位符方式, 避免SQL注入攻击问题.
            # 2. 获取查询结果.
            result = self.cursor.fetchone()         # fetchone(): 获取单行数据.
            # 3. 输出查询到的结果.
            # print(f'result: {result}')
            # 4. 处理结果: 有匹配数据则返回答案(取元组的第1个元素), 无匹配返回None
            return result[0] if result else None
        except pymysql.MySQLError as e:
            # 5. 输出错误信息.
            self.logger.error(f'答案获取失败: {e}')
            # 6. 返回空字符串, 避免异常.
            return None


    # todo 2.6 关闭MySQL连接 -> 释放资源, 避免连接泄漏.
    def close(self):
        # 关闭数据库连接
        try:
            # 关闭连接
            self.connection.close()
            # 记录关闭成功
            self.logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            self.logger.error(f"关闭连接失败: {e}")



