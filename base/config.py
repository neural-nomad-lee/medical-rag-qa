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
