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

from rag_qa.core.query_classifier import QueryClassifier

DATA_FILE = os.path.join(ROOT, "rag_qa", "classify_data", "medical_classify_300.json")
MODEL_DIR = os.path.join(ROOT, "rag_qa", "models", "bert_query_classifier")


def main():
    # 1. 用一个不存在的临时路径实例化 -> load_model() 会从 bert-base-chinese 基座
    #    初始化全新模型(而不是加载旧的教育版权重, 避免旧分布偏差残留).
    fresh = QueryClassifier(model_path=os.path.join(ROOT, "rag_qa", "models", "_tmp_fresh_init"))
    # 2. 训练完成后模型要保存到正式目录, 训练前把 model_path 指回正式路径.
    fresh.model_path = MODEL_DIR
    # 3. 用医疗版训练集训练(300条: 专业咨询/通用知识 各150, 80/20划分).
    fresh.train_model(DATA_FILE)
    print("训练完成, 模型已保存至:", MODEL_DIR)


if __name__ == "__main__":
    main()
