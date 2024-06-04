#!/usr/bin/env python3
# coding: utf-8
"""
init_milvus_data.py —— 一键将医疗知识库文档编码入库 Milvus

【作用】
  1. 连接 Milvus（medrag 库 / medrag_collection 集合，随 config.ini 配置）；
  2. 遍历 rag_qa/data/ 下与 config.ini [app].departments 对应的 {科室}_data 目录；
  3. 加载文档 → 父块/子块分层切分 → BGE-M3 编码 → 混合入库（稠密+稀疏向量）。

【说明】
  · 相同内容/相同问题(结构化条目)重复入库时按主键 upsert 覆盖，不会产生重复数据；
  · 本脚本为幂等设计，可重复执行。

【用法】(在项目根目录执行)
  python scripts/init_milvus_data.py
"""

import os
import sys
