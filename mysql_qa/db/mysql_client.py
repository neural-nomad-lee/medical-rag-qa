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
