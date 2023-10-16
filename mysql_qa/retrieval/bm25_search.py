# 该脚本用于: BM25搜索, 阈值: >= 0.85才会走MySQL系统.

# 导包
import os, sys                                  # 标准库：路径与模块搜索路径
# 把"项目根目录"加入 sys.path，这样直接用 `python mysql_qa/retrieval/bm25_search.py`
# 单独运行本文件时，也能解析 `from mysql_qa.xxx import ...` 这类绝对导入。
# （原文件缺这段，导致单独运行报 ModuleNotFoundError: No module named 'mysql_qa'）
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rank_bm25 import BM25Okapi                 # 导入 BM25 算法
import numpy as np                              # 导入数值计算库
from mysql_qa.utils.preprocess import preprocess_text    # 导入文本预处理
from mysql_qa.db.mysql_client import MySQLClient         # 导入MySQL数据库操作
from mysql_qa.cache.redis_client import RedisClient      # 导入Redis缓存操作
from base import logger                         # 导入日志


# todo 1. 定义BM25搜索类 -> 封装数据加载, 模型初始化, 相似度计算, 答案检索等...
class BM25Search:
    # todo 1.1 初始化函数 -> 客户端实例, 模型参数, 加载数据.
    def __init__(self, redis_client, mysql_client):
        # 1. 日志实例.
        self.logger = logger
        # 2. Redis客户端实例.
        self.redis_client = redis_client
        # 3. MySQL客户端实例.
        self.mysql_client = mysql_client
        # 4. 初始化BM25模型.
        self.bm25 = None
        # 5. 初始化分词后的问题列表.
        self.questions = None
        # 6. 初始化原始问题列表.
        self.original_questions = None
        # 7. 调用数据加载方法 -> 加载问题数据, 并初始化模型.
        self._load_data()       # 小细节: 加_的函数, 表示: 内置函数

    # todo 1.2 数据加载方法 -> 优先从Reids缓存加载, 缓存未命中则从MySQL加载.
    def _load_data(self):
        # 1. 定义Redis缓存键 -> 区分原始问题 和 分词后问题的缓存.
        original_key = "qa_original_questions"      # 原始问题缓存键, 它的数据格式为: ['问题1', '问题2', '问题3'...]
        tokenized_key = "qa_tokenized_questions"    # 分词后问题缓存键, 它的数据格式为: [['问题1切词后的词1', '词2', '词3'...], ['问题2切词词1', '词2'...], ['问题3...']...]

        # 2. 从Redis中获取原始问题 -> 缓存读取速度快, 减少数据库访问.
        self.original_questions = self.redis_client.get_data(original_key)
        # print(f'original_questions: {self.original_questions}')

        # 3. 从Redis中获取分词后的问题.
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        # print(f'tokenized_questions: {tokenized_questions}')

        # 4. 若Redis缓存中无数据(或者数据不完整) -> 则从MySQL加载并更新缓存.
        if not self.original_questions or not tokenized_questions:
            # 4.1 从MySQL获取原始文件列表, 格式为 元组嵌套, 例如: ((问题1, ), (问题2, )...)
            self.original_questions = self.mysql_client.fetch_questions()
            # print(f'original_questions: {self.original_questions}')

            # 4.2 若MySQL也无问题数据, 记录警告日志并返回.
            if not self.original_questions:
                self.logger.warning('未从MySQL加载到任何问题数据')
                return

            # 4.3 对原始问题进行分词处理 -> 转换为词语列表.
            # 注意: q[0]是因为原始问题格式为: (('问题1', ), ('问题2', )...)
            tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions]

            # 4.4 将原始问题列表转换为字符串 -> 去除元组嵌套, 并存换到Redis中.
            self.redis_client.set_data(original_key, [(q[0]) for q in self.original_questions])
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        # 5. 保存分词后的问题列表 -> 用于BM25模型初始化.
        self.questions = tokenized_questions
        # 6. 初始化BM25模型.
        self.bm25 = BM25Okapi(self.questions)
        # 7. 记录INFO日志
        self.logger.info('BM25模型 初始化成功')

    # todo 1.2.1 缓存刷新方法 -> 知识库更新后主动失效缓存并重新加载问题列表.
    def refresh(self):
        """
        函数功能: 清除Redis中的问题列表缓存, 并从MySQL重新加载, 重建BM25索引.
        使用场景: 新知识(RAG沉淀的问答)写入 ylkb 表后调用, 让 BM25 能立即检索到新问题.
        :return: True -> 刷新成功, False -> 刷新失败
        """
        try:
            # 1. 删除两类缓存键 -> 下次 _load_data() 会重新从MySQL加载并写回缓存.
            self.redis_client.delete_keys('qa_original_questions', 'qa_tokenized_questions')
            # 2. 重置内存状态并重新加载.
            self.original_questions = None
            self.questions = None
            self.bm25 = None
            self._load_data()
            # 3. 记录日志.
            self.logger.info('BM25 缓存刷新完成')
            return True
        except Exception as e:
            self.logger.error(f'BM25 缓存刷新失败: {e}')
            return False


    # todo 1.3 softmax分数归一化方法 -> 将BM25分数转换为概率分布, 便于阈值判断.
    def _softmax(self, scores):
        # 对输入分数进行softmax归一化, 输出总和为1的概率分布.
        # 1. 计算指数分数: 每个分数减去最大值 -> 防止数值过大导致溢出.
        exp_scores = np.exp(scores - np.max(scores))
        # 2. 返回归一化结果.
        return exp_scores / np.sum(exp_scores)


    # todo 1.4 核心搜索方法: 处理查询, 计算相似度, 返回匹配答案 或 提示无结果.
    def search(self, query, threshold=0.85):
        """
        根据输入查询检索 最相似的问题 并返回对应答案
        :param query: 用户查询文本
        :param threshold: 相似度阈值 -> 超过此值即为匹配成功
        :return: 匹配成功: (答案, False),  未匹配: (None, True)      True:新查询, False: 非新查询(命中缓存)
        """
        # 1. 检查查询有效性 -> 空或者非字符串类型认为无效.
        if not query or not isinstance(query, str):
