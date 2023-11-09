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
