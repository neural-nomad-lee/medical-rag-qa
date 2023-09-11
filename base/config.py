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
