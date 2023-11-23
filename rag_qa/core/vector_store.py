# 该脚本用于: 向量存储和检索.
# 核心流程: 文档向量生成(BGE-M3) -> 向量入库(Milvus) -> 混合检索(稠密 + 稀疏向量) -> 结果去重 -> 重排序 -> 返回精准文档.
# 常用的检索策略介绍:
#   1. 密集检索: 语义级匹配 -> '苹果关机'匹配'IPhone重启'
#   2. 稀疏检索: 关键词级过滤 -> 去合同中找'违约责任'
#   3. 多向量检索: 逐词细节对比 -> 区分'人工智能'和'AI'
# 好处: 覆盖'语义理解-快速过滤-细节对比'全场景, 适用于: 智能客服,法律文档定位,论文对比分析...

# BGE-M3模型介绍: 是一款超全能的文本'翻译器' + '搜索引擎', 能把文字转成词向量, 帮计算机快速理解和检索信息.


# 模型设备相关: 判断GPU是否可用, 为模型选择运行设备.
import torch.cuda
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库, 例如: 客户端, 数据类型, 检索请求, 排序器
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象, 即: 统一文档数据格式(含内容 + 元数据)
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断 -> 优化检索结果的相关性排序.
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值 -> 作为Milvus的主键.
import hashlib
import sys, os
# 说明：原代码是 `from document_processor import *`（裸导入），
# sys.path 里只有 rag_qa/ 而没有 rag_qa/core/，所以这条导入会抛
#   ModuleNotFoundError: No module named 'document_processor'
# 一旦被导入（rag_qa/__init__.py 会导入本模块），整个 `from rag_qa import ...` 全部失败。
# 注意：它在 vector_store.py 里其实只被注释掉的测试代码用到，
# 属于"没用的导入却能把整个包拖崩"——典型的隐蔽故障。
# 改成包内正确路径（rag_qa/ 已在 sys.path 中，见 rag_qa/__init__.py）：
from rag_qa.core.document_processor import process_documents, load_documents_from_directory


# todo 1. 路径配置模块. 定位当前脚本, RAG核心目录, 项目根目录, 确保自定义模块可导入.
# 1.获取当前文件所在目录的绝对路径.
local_path = os.path.abspath(os.path.dirname(__file__))
# print(f'local_path: {local_path}')      # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa\core

# 2. 获取当前目录的上级目录.
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
# print(f'module_dir: {rag_qa_path}')     # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa

# 3. 获取项目根目录的绝对路径.
project_root = os.path.dirname(rag_qa_path)
# print(f'project_root: {project_root}')  # D:\workspace\ai_30_bj\integrated_qa_system

# 4. 将项目根目录加入系统路径, 确保跨目录导入自定义模块(例如: config, logger)...
sys.path.insert(0, project_root)

from base.config import Config
from base.logger import logger


# todo 2.初始化全局配置 -> 实例化配置对象.
conf = Config()

# todo 3.定义VectorStore类: 封装向量库的核心操作 -> 集合管理, 文档入库, 混合检索, 结果处理.
class VectorStore:
    # todo 3.1 初始化方法 -> 配置向量库连接, 加载模型, 初始化客户端.
    def __init__(
        self,
        collection_name=conf.MILVUS_COLLECTION_NAME,        # Milvus集合名 -> 类似于SQL的表名, 这里我们叫: medrag_collection
        host=conf.MILVUS_HOST,                              # Milvus主机地址
        port=conf.MILVUS_PORT,                              # Milvus端口
        database=conf.MILVUS_DATABASE_NAME                  # Milvus数据库名
    ):
        # 1. 存储Milvus的核心设置.
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database

        # 2. 初始化日志实例.
        self.logger = logger

        # 3. (扩展)选择模型运行设备 -> 优先使用GPU, 无GPU则用CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'使用设备: {self.device}')

        # 4. 初始化BGE-Reranker重排序模型 -> 优化检索结果相关性排序.
        # 4.1 拼接重排序模型的本地路径. 即: rag_qa/models/bge-reranker-large
        reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        # print(f'reranker_path: {reranker_path}')

        # 4.2 加载模型 -> 指定运行设备,  模型用于计算'查询-文档'的相关性得分.
        self.reranker = CrossEncoder(reranker_path, device=self.device)

        # 5. 初始化BGE-M3模型 -> 用于生成文档和查询的向量表示.
        # 5.1 拼接模型文件的本地路径. 即: rag_qa/models/bge-m3
        m3_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
        # 5.2 加载模型 -> 模型用于生成文档和查询的向量表示.
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,         # 模型本地路径
            use_fp16=(self.device == 'cuda'),   # GPU时启用半精度计算(减少内存占用, 提升速度), CPU时禁用
            device=self.device)

        # 6. 获取BGE-M3稠密向量的维度: 固定输出1024维稠密向量.
        self.dense_dim = self.embedding_function.dim['dense']
        # print(f'稠密向量的维度: {self.dense_dim}')  # 结果: 1024

        # 7. 初始化Milvus客户端, 建立: Milvus向量库连接.
        # uri格式: http://主机地址:端口号, db_name 指定要连接的数据库.
        # 【自动建库】全新 Milvus 实例不会自动创建业务库, 先在 default 库上检查并创建,
        # 避免连接报 MilvusException: database not found.
        _bootstrap = MilvusClient(uri=f"http://{self.host}:{self.port}")
        if self.database not in _bootstrap.list_databases():
            _bootstrap.create_database(self.database)
            logger.info(f"已创建 Milvus 数据库: {self.database}")
        _bootstrap.close()
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)

        # 8. 调用'私有'方法 -> 创建新集合(若不存在) 或 加载已有集合(若存在)
        self._create_or_load_collection()


    # todo 3.2 私有方法: 创建新集合(若不存在) 或 加载已有集合(若存在)
    def _create_or_load_collection(self):
        # 1. 检查指定集合是否已存在(避免重复创建)
        if not self.client.has_collection(self.collection_name):
            # 走这里, 说明集合不存在, 需要创建集合.
            # 2. 创建集合 Schema，禁用自动 ID(手动用文档哈希作为主键)，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加科室类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 3.创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}        # 构建索引时,丢弃20%的低权重值(减少存储,噪声, 精度影响较小)
            )

