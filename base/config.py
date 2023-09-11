# 该脚本用于: 配置文件管理.

# 导包
import configparser                 # 导入配置文件解析库, 解析INI格式的配置文件.
import os                           # 导入路径操作模块


# todo 1.配置文件路径计算.
# 1. 获取当前文件的绝对路径.
current_file_path = os.path.abspath(__file__)
# print(f'♥ current_file_path: {current_file_path}')  # C:\project_code\integrated_qa_system\base\config.py
# 2. 获取当前文件所在目录的绝对路径.
current_dir_path = os.path.dirname(current_file_path)
# print(f'current_dir_path: {current_dir_path}')      #C:\project_code\integrated_qa_system\base

# 3. 获取项目根目录的绝对路径
project_root = os.path.dirname(current_dir_path)
# print(f'project_root: {project_root}')                #C:\project_code\integrated_qa_system

# 4. 拼接配置文件(config.ini)的完整路径.
config_file_path = os.path.join(project_root, 'config.ini')
# print(f'config_file_path: {config_file_path}')          #C:\project_code\integrated_qa_system\config.ini


# todo 2.配置解析类 -> 封装配置文件读取逻辑, 提供统一的配置参数访问接口.
class Config:
    # todo 2.1 初始化方法: 读取配置文件并解析各服务的配置参数.
    def __init__(self, config_file=config_file_path):
        # # 1. 创建配置文件解析器实例.
        # self.config = configparser.ConfigParser()
        # # 2. 读取配置文件 -> 根据传入的路径加载ini文件内容.
        # self.config.read(config_file, encoding='utf-8')
        #
        # # 3. 解析并存储各服务配置参数.
        # # 3.1 解析MySQL数据库配置.
        # self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        # self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        # self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='123456')
        # self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='med_kg')
        #
        # # 3.2 解析Redis数据库配置.
        # self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        # self.REDIS_PORT = self.config.get('redis', 'port', fallback=6379)
        # self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='1234')
        # self.REDIS_DB = self.config.get('redis', 'db', fallback=0)
        #
        # # 3.3 解析日志配置.
        # # 日志文件路径: 从 [logger]节点的log_file键获取, 默认值为: logs/app.log
        # self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')


        # 创建配置解析器，启用插值功能
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        # 如果没有提供配置文件路径，则使用默认路径
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        # 【安全】加载项目根目录 .env 中的敏感配置(如 API Key):
        #   · load_dotenv 只在环境变量不存在时才写入, 已有系统环境变量优先;
        #   · 未安装 python-dotenv 时静默跳过, 退回 config.ini fallback(此时 key 需手工配置).
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(self.PROJECT_ROOT, '.env'))
        except ImportError:
            pass

        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        self.MED_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/med_document_loaders')

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        # 读取配置文件
        self.config.read(config_file, encoding='utf-8')

        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        # MySQL 用户名
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
