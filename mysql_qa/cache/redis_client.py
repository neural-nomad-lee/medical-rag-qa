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
# sys.path.insert(0, project_root)

# 5. 导入自定义配置 和 日志模块.
from base import Config, logger


# todo 1.定义Redis客户端类 -> 封装Redis连接, 数据存储, 数据获取, 答案查询等功能.
# 【过期策略】所有写入 Redis 的数据统一设置 20 小时(72000秒)过期:
#   1. 避免问答缓存/BM25问题缓存无限累积, 占用内存.
#   2. MySQL 知识库更新后, 最多 20 小时内缓存自动失效并重新加载, 保证数据一致性.
class RedisClient:
    # 数据默认过期时长: 20小时 = 20 * 3600 = 72000 秒
    DEFAULT_EXPIRE_SECONDS = 20 * 3600
    # todo 1.1 初始化方法, 获取Redis连接, 初始化日志, 处理连接异常.
    def __init__(self):
        # 1. 获取日志记录器.
        self.logger = logger

        # 2. 建立Redis连接.
        try:
            self.client = redis.StrictRedis(
