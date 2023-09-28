# 该脚本用于: Redis缓存技术.

# 导包
import redis
import json
import os, sys

# todo 0.路径相关 -> 如下的设置是为了 <<将项目根目录加入系统路径>>, 绝大多数电脑不设置也可以, 如果你直接导包遇到问题了, 请执行如下设置.
# # 1.获取当前文件所在目录的绝对路径.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir: {current_dir}')
#
# # 2. 获取当前目录的上级目录.
# module_dir = os.path.dirname(current_dir)
# print(f'module_dir: {module_dir}')
#
# # 3. 获取项目根目录的绝对路径.
# project_root = os.path.dirname(module_dir)
# print(f'project_root: {project_root}')
#
# # 4. 将项目根目录加入系统路径, 确保跨目录导入自定义模块(例如: config, logger)...
