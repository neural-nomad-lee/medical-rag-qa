# 该脚本核心功能: 统一管理RAG流程中所需的各类Prompt模板.
# 作用: 通过LangChain#PromptTemplate创建 Prompt 模板，将不同场景的提示词(如: 直接检索, 子查询, 回溯问题)，并返回最终的 Prompt 模板.
#      后续只需传入具体参数(例如: 上下文, 问题)即可快速生成符合需求的提示词, 避免重复编写提示词.


# 导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain_core.prompts import PromptTemplate


# todo 1.定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    # todo 1.1 定义 RAG 提示模板 -> 根据上下文生成答案, 无上下文则用自身知识, 无法回答时返回: 客服信息.
    @staticmethod
    def rag_prompt():
        """
        RAG 主生成模板。

        ★ 为什么必须保留 {history} 占位符？（这是本次报错的根因）
          两个调用方传的参数不一样：
            new_rag_system.py（V2，rag_qa/__init__.py 实际启用的）传 4 个：context, history, question, phone
            rag_system.py（V1，旧版）                      传 3 个：context, question, phone
