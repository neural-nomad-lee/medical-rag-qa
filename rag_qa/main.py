# 该脚本是: RAG_QA系统的主入口.

import os
import sys
from base.config import config as conf
from base.logger import logger
from rag_qa.core.document_processor import process_documents  # 导入处理文档的函数
from rag_qa.core.vector_store import VectorStore              # 向量存储和检索
from rag_qa.core.new_rag_system import RAGSystem              # RAG_QA系统的核心代码(V2: 含历史对话与流式输出)
from openai import OpenAI                                     # 使用 OpenAI 接口


# todo 1.定义主函数main -> 系统入口逻辑, 根据模式执行数据处理或交互式查询.
def main(query_mode=True, directory_path="data"):
    """
    函数功能: 系统主函数, 控制两种运行模式.
    :param query_mode:
        数据处理模式: query_model=False  解析指定目录的文档, 分割为文档块并存入向量数据.
        交互式查询模式: query_model=True 接收用户问题, 调用RAG系统生成答案并展示.
    :param directory_path: 数据处理模式下的文档根目录, 默认是: ./data, 需包含各个科室的子目录, 例如: 心血管内科_data, 呼吸内科_data
    :return: 无
