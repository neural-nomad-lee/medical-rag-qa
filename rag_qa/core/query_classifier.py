# 该脚本用于: 基于BERT模型实现查询分类器, 将用户查询分为'通用知识'和'专业咨询'两大类.
# 核心流程: 加载预训练BERT模型 -> 数据预处理 -> 模型训练 -> 模型评估 -> 预测查询类别
# 应用场景: 在RAG系统中, 可根据分类结果决定是否需要检索专业文档(例如: 专业咨询可检索文档, 通用知识由模型直接回复即可)


# todo 1.导包
# 导入标准库
import json
import os, sys

# 路径配置: 解决跨目录模块导入问题 -> 定位项目各层级目录.
current_dir= os.path.dirname(os.path.abspath(__file__))   # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa\core
rag_qa_path = os.path.dirname(current_dir)      # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa
project_root = os.path.dirname(rag_qa_path)     # D:\workspace\ai_30_bj\integrated_qa_system
# 将项目目录添加到系统路径中
sys.path.insert(0, project_root)

# 导入 PyTorch
import torch
# 导入日志
from base.logger import logger
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification       # BERT分词器和序列分类模型.
# Trainer: 负责实际执行训练过程(用模型, 数据和 TrainingArguments的配置跑训练) -> 相当于 训练任务的执行者, 按照 TrainingArguments(说明书)具体干活
# TrainingArguments: 负责定义训练的各种参数和配置(例如: 训练多久, 多大批次, 保存到哪等...) -> 相当于 训练任务的说明书,提前写好训练的各种规则和细节.
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# todo 2.定义QueryClassifier类: 封装BERT查询分类的完整流程 -> 模型加载, 训练, 评估, 预测...
class QueryClassifier:
    # todo 2.1 初始化方法: 配置模型路径, 加载分词器, 选择设备, 定义标签映射.
    def __init__(self, model_path=None):
        # 【修复】原默认值是相对路径 '../models/bert_query_classifier'，依赖 cwd。
        # 改为：传 None 时用 rag_qa_path 推导绝对路径，从任何目录运行都正确。
        # 1. 存储模型路径: 用于后续加载或保存模型.
        self.model_path = model_path or os.path.join(rag_qa_path, 'models', 'bert_query_classifier')
