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

