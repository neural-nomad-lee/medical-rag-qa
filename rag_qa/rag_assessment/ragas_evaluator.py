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
