# 自定义的 日志工具类.

# 导包
import logging
import os
from config import Config


# todo 1.计算日志文件的最终保存路径.
# 1. 获取当前脚本文件的绝对路径.
module_path = os.path.abspath(__file__)
print(f'module_path: {module_path}')        #  C:\project_code\integrated_qa_system\base\logger.py

# 2. 获取当前脚本所在目录的路径.
base_path = os.path.dirname(module_path)
print(f'base_path: {base_path}')              #  C:\project_code\integrated_qa_system\base

# 3. 获取当前脚本所在目录的父目录的路径.
current_path = os.path.dirname(base_path)
print(f'current_path: {current_path}')          #  C:\project_code\integrated_qa_system

# 4. 拼接日志文件的完整路径.   上级目录 + logs文件夹 + app.log文件.
# log_file = os.path.join(current_path, 'logs/app.log')
