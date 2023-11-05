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
          如果模板里没有 {history}：
            · V2 调用时，传进去的 history 会被静默忽略 → 多轮对话能力消失（不报错，最难查）
          如果模板里有 {history} 但 input_variables 漏写：
            · LangChain 构造时会警告，format 时报 KeyError
          正确做法 = 模板含 {history} + input_variables 声明它。
          V1 不传 history 也不会报错：PromptTemplate 的 format 允许传入多余的 kwargs，
          只要求 input_variables 里的变量都提供；而 {history} 在 V1 场景下会渲染成空字符串。
        """
        return PromptTemplate(
            template="""
                你是一个智能助手，帮助用户回答问题。请参考对话历史和上下文回答问题。
                如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
                如果答案来源于检索到的文档，请基于上下文回答，并在回答中使用 [1]、[2] 等编号
                标注所依据的上下文段落（上下文中每个段落以 [引用n] 开头，n 即编号）。

                对话历史: {history}
                上下文: {context}
                问题: {question}

                如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
                回答:
