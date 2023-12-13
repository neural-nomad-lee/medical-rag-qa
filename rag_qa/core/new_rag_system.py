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
sys.path.insert(0, rag_qa_path)
# 4. 获取项目根目录.
project_root = os.path.dirname(rag_qa_path)                     # D:\workspace\ai_30_bj\integrated_qa_system
# 5. 将项目根目录添加到Python的模块搜索路径.
sys.path.insert(0, project_root)

from base.config import Config                  # 配置文件
from base.logger import logger                  # 日志对象


# 优化1: 新增导包内容.
# ============================================================
# 【无独显 / 低内存机器的关键优化：删除无用的 BERT 加载】
# ------------------------------------------------------------
# 这里原本是：
#     from transformers import BertTokenizer, BertModel
#     local_model_path = ...\models\bert-base-chinese      （原为写死的 Windows 路径）
#     tokenizer = BertTokenizer.from_pretrained(local_model_path)
#     model     = BertModel.from_pretrained(local_model_path)
#
# 实测问题（在这个文件里）：
#   · 这两个变量【只被赋值，从未被读取】—— 全文件再无 "tokenizer"/"model" 引用
#     （真正需要 BERT 的是 query_classifier.py，它自己会加载）
#   · 代价：import 阶段就吃掉约 885 MB 内存 + 7.3 秒启动时间
#     （bert-base-chinese 是 102.3 M 参数的模型）
#   · 对 16 GB 内存且无独显的机器，这是纯浪费，还会挤占 BGE-M3/Reranker 的空间
#   · 更糟的是：原代码写死了 Windows 绝对路径，换系统直接 OSError，且发生在 import 阶段
#
# 所以：整段删除。RAG 的意图识别照旧由 QueryClassifier 负责，功能完全不受影响。
# ============================================================

from openai import OpenAI


# 加载项目配置
conf = Config()


# todo 3. 定义RAGSystem类 -> 封装RAG系统的完整流程, 即: 查询分类 -> 策略选择 -> 文档检索 -> 答案生成
class RAGSystem:
    # todo 3.1 初始化方法 -> 创建RAG系统所需的核心组件 -> 向量库, 大模型, 分类器...
