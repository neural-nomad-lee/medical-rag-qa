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
