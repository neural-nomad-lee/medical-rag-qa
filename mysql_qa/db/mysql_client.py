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
