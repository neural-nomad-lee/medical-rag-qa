#!/usr/bin/env python3
# coding: utf-8
"""
generate_medical_classify_data.py —— 生成医疗版 BERT 查询分类训练数据

【用途】
  原训练集 model_generic_5000.json 全部为教育领域语料（课程/学费/大纲等），
  医疗问答场景下会把大量医疗问题误判为"通用知识"，绕过 RAG 检索链路。
  本脚本按模板组合生成"通用知识 / 专业咨询"两类均衡、多样的医疗版训练集。
  (数据量已按需求由 5000 条缩减为 300 条)

【输出】
  rag_qa/classify_data/medical_classify_300.json (JSONL, 每行: {"query": ..., "label": ...})

【用法】
  python scripts/generate_medical_classify_data.py [生成条数]   # 默认 300
"""

import json
import os
