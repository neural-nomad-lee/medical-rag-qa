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
        # 2. 加载BERT分词器: 将文本转换成模型可以理解的输入.
        # 2.1 拼接预训练BERT模型的本地路径（★存成实例属性，供 load_model() 使用）
        self.bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
        # 2.2 加载分词器.
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        # 3. 初始化模型变量, 后续通过 load_model()加载或创建模型.
        self.model = None

        # 4. 选择模型运行设备.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 5. 记录日志.
        logger.info(f'使用设备: {self.device}')

        # 6. 定义标签映射: 将标签映射为模型输出的索引. 即: 通用知识 -> 0, 专业咨询 -> 1.
        self.label_map = {"通用知识": 0, "专业咨询": 1}

        # 7. 加载模型: 初始化时, 自动调用load_model(), 确保模型可用.
        self.load_model()


    # todo 2.2 加载模型方法 -> 从指定路径加载已训练模型, 若不存在则初始化新模型.
    def load_model(self):
        """
        函数功能: 加载模型, 优先从self.model_path加载, 不存在则初始化基于 bert-base-chinese的新模型
        :return:
        """
        # 1. 检查模型路径是否存在 -> 即: 是否有已训练的模型.
        if os.path.exists(self.model_path):
            # 加载预训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录加载成功的日志
            logger.info(f"加载模型: {self.model_path}")
        else:
            # 2. 若模型不存在, 初始化新模型
            # 参1: 模型路径,  参2: 二分类任务.
            # 【修复】原代码此处是相对路径 "../models/bert-base-chinese"，
            # 只在 cwd = rag_qa/ 时有效；从项目根或其他目录运行就 OSError。
            # 改用上面已算好的绝对路径 bert_path（与 self.tokenizer 的加载路径一致）
            self.model = BertForSequenceClassification.from_pretrained(self.bert_path, num_labels=2)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录初始化模型的日志
            logger.info("初始化新 BERT 模型")


    # todo 2.3 保存模型的方法 -> 将训练好的模型和分词器存储到本地.
    def save_model(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"保存模型至: {self.model_path}")


    # todo 2.4 数据预处理方法 -> 将文本和标签转换为BERT模型所需的输入格式.
    def preprocess_data(self, texts, labels):
        """
        函数功能: 对文本进行分词, 阶段, 填充, 将标签转换为: 数字.
        :param texts: 待处理的文本列表.
        :param labels: 文本对应的标签列表.
        :return: 处理后的编码 (input_ids, attention_mask) 和 数字标签列表.
        """
        # 1. 文本编码: 使用BERT分词器将文本转换为模型输入格式.
        encodings = self.tokenizer(
            texts,
            truncation=True,        # 超过最大长度时截断
            padding=True,           # 不足最大长度时填充.
            max_length=128,         # 最大长度
            return_tensors="pt"     # 返回张量
        )
        # 2. 标签转换: 将文本标签转换为数字
        return encodings, [self.label_map[label] for label in labels]


    # todo 2.5 创建数据集方法: 自定义Dataset类, 封装: 编码 和 标签.
    def create_dataset(self, encodings, labels):
        # 1. 定义内部Dataset类, 继承子torch.utils.data.Dataset
        class Dataset(torch.utils.data.Dataset):
            # 2. 初始化方法: 接收编码和标签, 创建内部Dataset类.
            def __init__(self, encodings, labels):
                super().__init__()
                self.encodings = encodings  # 文本编码, 格式: (input_ids, attention_mask)
                self.labels = labels        # 数字标签列表, 格式: [0, 1, 0, 1, ...]

            # 3. 根据索引返回单条数据( 编码 + 标签)
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            # 4. 返回数据集长度.
            def __len__(self):
                return len(self.labels)

        # 5. 返回实例化的Dataset对象.
        return Dataset(encodings, labels)


    # todo 2.6 训练模型方法 -> 加载数据集, 预处理, 配置训练参数并训练模型.
    def train_model(self, data_file="medical_classify_300.json"):
        """
        函数功能: 训练BERT分类模型, 区分查询分类为'通用知识' 和 '专业咨询'
        :param data_file: 数据集文件路径.
        :return: 无
        """
        # 1. 检查数据集文件是否存在.
        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        # 2. 加载数据集, 从Json文件中读取查询文本和对应标签.
        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(value) for value in f.readlines()]

        # 3. 提取文本和标签, 分离查询文本和对应的分类标签.
        texts = [item["query"] for item in data]        # 格式: [查询文本1, 查询文本2, 查询文本3, ...]
        labels = [item["label"] for item in data]       # 格式: [标签1, 标签2, 标签3, ...]

        # 4. 划分训练集和测试集. 80%用于训练, 20%用于验证. 固定随机种子确保结果可复现.
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # 5. 数据预处理: 将文本和标签转换为BERT模型所需的输入格式 -> 调用自定义的preprocess_data() 预处理数据的方法.
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)

        # 6. 创建数据集对象, 将编码 和 标签封装为PyTorch可识别的 Dataset对象.
        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)


        # 7. 配置训练参数: 定义模型训练的超参数.
        training_args = TrainingArguments(
            # 设置模型和检查点保存的目录路径
            output_dir="./bert_results",
            # 设置训练的总轮数为3轮
