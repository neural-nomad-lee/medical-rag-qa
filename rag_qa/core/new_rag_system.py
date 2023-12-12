# 该脚本用于: RAG系统的核心逻辑 -> 实现RAG(检索增强生成)系统的核心逻辑, 整合查询分类(意图识别), 策略选择, 文档检索 和 答案生成.
# 注: 该脚本的 generate_answer 方法较之于旧版做了优化, 加入了 history历史对话.


# todo 1.导包
import sys, os                                  # 导入路径处理和系统配置.
import time                                     # 导入 time 模块，用于计算时间

from rag_qa.core.prompts import RAGPrompts                  # 导入RAG相关的提示模板 -> 定义大模型的输入和输出格式.
from rag_qa.core.query_classifier import QueryClassifier    # 导入查询分类器 -> 判断用户问题属于通用知识还是专业咨询,意图识别
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器 -> 专业咨询的情况下用哪种检索策略(例如:直接检索...)
from rag_qa.core.vector_store import VectorStore            # 导入向量数据库对象 -> 存储和检索文档向量.


# todo 2.配置项目路径 -> 确保程序能正确导入其它目录的模块.
# 1. 获取当前文件的目录的绝对路径.
current_dir = os.path.dirname(os.path.abspath(__file__))        # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa\core
# 2. 获取当前目录的上级目录.
rag_qa_path = os.path.dirname(current_dir)                      # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa
# 3. 将rag_qa目录添加到Python的模块搜索路径.
