# coding: utf-8
"""
ragas_evaluator.py —— 基于 RAGAS 框架的 RAG 系统端到端评估

【评估什么】
  不是拿"预先写死的问答数据"打分, 而是把评估集里的真实问题喂给系统,
  走完 检索(Milvus混合检索+重排) -> 生成(LLM流式回答) 的完整链路,
  再用 RAGAS 对"检索质量"和"生成质量"分别打分:

  检索部分:
    · context_precision  上下文精确率: 检回的上下文里, 相关内容是否排在前面(没夹带无关块)
    · context_recall     上下文召回率: 回答问题所需的信息, 是否都被检索回来了
  生成部分:
    · faithfulness       忠实度: 答案是否严格基于检索到的上下文(衡量幻觉程度, 越接近1越好)
    · answer_relevancy   答案相关性: 答案是否切题(有没有答非所问)

【评估用的模型】
    · LLM:       复用系统的 DashScope 兼容端点(qwen-plus, 密钥从 .env 读取)
    · Embeddings: 复用系统已加载的本地 BGE-M3(不额外调API, 零成本)

【数据格式】ragas_eval_data.json: [{"question": ..., "ground_truth": ...}, ...]
  ground_truth 为人工整理的标准答案, 与 rag_qa/data/ 知识库内容对齐。

【用法】(在项目根目录执行, 需先启动 Milvus 容器并完成知识入库)
  python rag_qa/rag_assessment/ragas_evaluator.py
"""
import json
import os
import sys

# 1. 把项目根目录加入 sys.path, 保证从任何目录运行都能导入项目模块.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# todo 1.导包
from langchain_core.embeddings import Embeddings          # LangChain 嵌入模型接口(ragas 需要该接口)
from langchain_openai import ChatOpenAI                   # OpenAI 兼容协议的聊天模型(对接 DashScope 端点)
from datasets import Dataset                              # ragas 要求的数据集格式
from ragas import evaluate                                # ragas 评估入口
from ragas.metrics import (
    faithfulness,          # 忠实度
    answer_relevancy,      # 答案相关性
    context_precision,     # 上下文精确率
    context_recall,        # 上下文召回率
)

from base.config import Config
from base.logger import logger
from rag_qa.core.vector_store import VectorStore                    # 检索器(含 BGE-M3/Reranker)
from rag_qa.core.new_rag_system import RAGSystem                    # RAG 生成链路
from rag_qa.core.strategy_selector import StrategySelector          # 复用其 LLM 调用函数


# todo 2.定义评估用的嵌入模型适配器 -> 把系统内已加载的 BGE-M3 适配成 ragas 需要的 LangChain Embeddings 接口.
# 好处: 不重复加载模型, 也不额外调用付费 Embedding API.
class BGEM3LangchainEmbeddings(Embeddings):
    """把 BGEM3EmbeddingFunction(FlagEmbedding) 包装成 LangChain Embeddings 接口."""

    def __init__(self, bgem3_function):
        # 传入 VectorStore 内已初始化好的 self.embedding_function 实例.
        self._bgem3 = bgem3_function

    def embed_documents(self, texts):
        """批量文本向量化(ragas 计算 answer_relevancy 时用于生成答案的相似问题/相似度)."""
        result = self._bgem3.encode_documents(list(texts))
        # 返回稠密向量列表(numpy 2D数组 -> list[list[float]]).
        return [vec.tolist() for vec in result["dense"]]

    def embed_query(self, text):
        """单条查询向量化."""
        result = self._bgem3.encode_queries([text])
        return result["dense"][0].tolist()


# todo 3.加载评估数据集 -> 每条含: question(问题) 与 ground_truth(人工整理的标准答案).
data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ragas_eval_data.json")
with open(data_file, "r", encoding="utf-8") as f:
    eval_cases = json.load(f)
logger.info(f"加载评估样本: {len(eval_cases)} 条")
for item in eval_cases:
    # 简单校验: 数据集必须同时具备 question 与 ground_truth 两个字段.
    assert "question" in item and "ground_truth" in item, f"数据缺字段: {item}"

# todo 4.初始化被评估的 RAG 系统(检索 + 生成), 并跑通完整链路收集 (question, answer, contexts).
# 4.1 实例化向量库 -> 加载 BGE-M3 与 Reranker(首次运行耗时较长).
vector_store = VectorStore()

# 4.2 实例化 RAG 系统 -> 与线上问答使用完全相同的检索/生成/分类链路.
llm_func = StrategySelector().call_dashscope
rag_system = RAGSystem(vector_store, llm_func)

questions, answers, contexts_list = [], [], []
for item in eval_cases:
    query = item["question"]
    logger.info(f"评估运行: {query}")
    # 4.3 走完整 RAG 链路生成答案(生成器, 逐段收集).
    answer = ""
    for token in rag_system.generate_answer(query):
        answer += token
    # 4.4 单独调用检索, 拿到参与生成的 Top-M 父文档作为 contexts(评估检索质量的依据).
    docs = vector_store.hybrid_search_with_rerank(query, source_filter=None)
    contexts = [doc.page_content for doc in docs]

    questions.append(query)
    answers.append(answer)
    contexts_list.append(contexts)
    logger.info(f"完成: 检索到 {len(contexts)} 个上下文块, 答案长度 {len(answer)} 字")

# todo 5.组装 RAGAS 数据集 -> 字段名固定: question / answer / contexts / ground_truth.
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts_list,
    "ground_truth": [item["ground_truth"] for item in eval_cases],
})
print(dataset)

# todo 6.配置评估模型 -> LLM 用系统同款端点(qwen-plus), Embeddings 复用本地 BGE-M3.
conf = Config()
eval_llm = ChatOpenAI(
