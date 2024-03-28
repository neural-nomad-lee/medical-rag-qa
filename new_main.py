# 该脚本是基于 old_main.py做的优化, 在其基础上添加了: 历史对话 + 流式输出.
# 你可以把这个版本理解为: 融合了MySQL的RAG系统 V2版本
# V2版本只需要优化3个地方:
#   1. 在 integrated_qa_system项目目录下创建 new_main.py 文件.
#   2. 优化 rag_qa/core/rag_system.py的代码, 建议拷贝一份然后改名叫 new_rag_system.py, 然后在其中修改即可.
#   3. 优化 rag_qa/__init__.py的代码, 导包从rag_system 改为 new_rag_system.

import warnings
warnings.filterwarnings("ignore")

# todo 1. 导入所需的库.
# 导入 MySQL 系统组件，用于数据库操作和搜索
from mysql_qa import MySQLClient, RedisClient, BM25Search
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa import VectorStore, RAGSystem

# 导入配置和日志工具，用于系统配置和日志记录
from base.config import config
from base.logger import logger
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI
