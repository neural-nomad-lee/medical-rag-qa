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
    """

    # 1. 初始化LLM客户端 -> 创建调用大模型的客户端,生成答案用.
    try:
        client = OpenAI(api_key=conf.DASHSCOPE_API_KEY, base_url=conf.DASHSCOPE_BASE_URL)
    except Exception as e:
        logger.error(f"初始化 OpenAI 客户端失败 (请检查 API Key 和 Base URL): {e}")
        # 如果客户端初始化失败，可能无法继续，取决于模式
        if query_mode: # 查询模式下必须要有 LLM
             print("错误：无法初始化语言模型客户端，无法进入查询模式。")
             return
        # 数据处理模式可能不需要 LLM，可以继续，但最好记录错误
        client = None # 标记客户端不可用


    # 2. 定义 LLM 调用函数 (仅在需要时定义和使用) -> 封装大语言模型调用逻辑, 供RAG 系统使用.
    def call_dashscope(prompt):
        if not client: # 检查客户端是否可用
            logger.error("LLM 客户端未初始化，无法调用 call_dashscope")
            return f"错误: LLM客户端不可用"
        try:
            completion = client.chat.completions.create(
