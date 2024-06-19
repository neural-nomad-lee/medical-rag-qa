# coding: utf-8
"""
train_medical_classifier.py —— 用医疗版训练集重训 BERT 查询分类器

【为什么要重训】
  rag_qa/models/bert_query_classifier 内的权重是用教育领域语料训练的,
  医疗问题会被大量误判为"通用知识"而绕过 RAG 检索链路。
  本脚本从 bert-base-chinese 基座(非旧权重)出发, 用 medical_classify_300.json
  全新训练二分类模型, 并覆盖保存到正式模型目录。

【用法】(在项目根目录执行)
  python scripts/train_medical_classifier.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

