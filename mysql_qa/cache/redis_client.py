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
                host=Config().REDIS_HOST,           # Redis数据库主机地址
                port=Config().REDIS_PORT,           # Redis数据库端口号
                password=Config().REDIS_PASSWORD,   # Redis数据库密码
                db=Config().REDIS_DB,               # Redis数据库编号
                decode_responses=True,              # 自动将返回的字节数据转换为字符串(避免手动解码)
                # 【连接健壮性】Redis 服务端会回收空闲连接(idle timeout), 连接池里的旧连接随之失效,
                # 再次使用时报 "Connection closed by server."。以下三项开启健康检查与自动重连:
                socket_keepalive=True,              # TCP 保活, 防止空闲连接被网络设备回收
                health_check_interval=30,           # 连接空闲超过30秒后, 复用前先 PING, 失效则自动重建
                retry_on_timeout=True               # 命令超时自动重试一次
            )
        except redis.RedisError as e:
            self.logger.error(f'Redis连接异常: {e}')
            raise   # 声明抛出异常.

        # 3. 记录INFO日志
        self.logger.info('Redis连接成功')

    # todo 1.2 存储数据到Redis -> 将Python对象序列化为json字符串存储(默认20小时过期).
    def set_data(self, key, value, expire_seconds=DEFAULT_EXPIRE_SECONDS):
        try:
            # 1. 存储数据, 并设置过期时间(20小时后自动删除).
            self.client.set(
                key,
                json.dumps(value, ensure_ascii=False),  # ensure_ascii: 保证中文等非ASCII字符正常显式.
                ex=expire_seconds                        # ex: 过期秒数, 到期自动删除.
            )
            # 2. 记录INFO日志.
            self.logger.info(f'Redis存储数据成功: {key} (过期时间: {expire_seconds}秒)')
        except redis.RedisError as e:
            self.logger.error(f'Redis存储数据失败: {e}')


    # todo 1.3 从Redis中获取数据 -> 反序列化为Python对象.
    def get_data(self, key):
        try:
            # 1. 获取数据.
            data = self.client.get(key)
            # 2. 处理结果 -> 反序列化为Python对象, 再返回.
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'Redis获取数据失败: {e}')
