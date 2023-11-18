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
