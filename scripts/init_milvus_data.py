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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# 复用项目配置与组件（与业务代码读同一份 config.ini）
from base.config import config
from base.logger import logger
from rag_qa.core.document_processor import process_documents
from rag_qa.core.vector_store import VectorStore


def main():
    data_root = os.path.join(ROOT, "rag_qa", "data")
    print("=" * 60)
    print(f"医疗知识库入库 | Milvus: {config.MILVUS_HOST}:{config.MILVUS_PORT} "
          f"| 库: {config.MILVUS_DATABASE_NAME} | 集合: {config.MILVUS_COLLECTION_NAME}")
    print("=" * 60)

    # 1. 初始化向量库（自动加载 BGE-M3 / Reranker 模型，首次运行耗时较长）
    vector_store = VectorStore(
        collection_name=config.MILVUS_COLLECTION_NAME,
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT,
        database=config.MILVUS_DATABASE_NAME,
    )

    # 2. 遍历科室目录, 处理入库
    total_chunks = 0
    for dept in config.DEPARTMENTS:
        dir_path = os.path.join(data_root, f"{dept}_data")
        if not os.path.isdir(dir_path):
            logger.warning(f"目录不存在，跳过: {dir_path}")
            print(f"  [跳过] {os.path.relpath(dir_path, ROOT)}（目录不存在）")
            continue

        print(f"  [处理] {os.path.relpath(dir_path, ROOT)} ...")
        chunks = process_documents(
            dir_path,
            config.PARENT_CHUNK_SIZE,
            config.CHILD_CHUNK_SIZE,
            config.CHUNK_OVERLAP,
        )
        if not chunks:
            logger.info(f"目录 {dir_path} 未发现有效文档")
            print("         未发现有效文档")
            continue

        vector_store.add_documents(chunks)
        total_chunks += len(chunks)
        print(f"         入库 {len(chunks)} 个子块")

    print("=" * 60)
    print(f"✅ 完成：共入库 {total_chunks} 个文档子块")


if __name__ == "__main__":
    main()
