#!/usr/bin/env python3
# coding: utf-8
"""
milvus_expiry_manager.py —— Milvus 知识库到期数据管理（一年期）

【策略】
  · Milvus 中的资料保存 1 年；到期后"先弹出提示、人工确认后"才执行删除；
  · 数据的入库时间记录在集合的 timestamp 字段（ISO 格式字符串）；
  · 确认删除只针对超期条目，未到期数据不受影响。

【用法】(在项目根目录执行)
  python scripts/milvus_expiry_manager.py            # 列出到期条目并交互确认删除
  python scripts/milvus_expiry_manager.py --list     # 只列出到期条目，不删除
  python scripts/milvus_expiry_manager.py --days 365 # 自定义到期天数，默认 365
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pymilvus import MilvusClient

from base.config import config
from base.logger import logger


def parse_ts(ts: str) -> datetime:
    """解析 Milvus 中存储的 ISO 时间字符串，解析失败返回最小时间(视为已到期，需人工确认)."""
    try:
        return datetime.fromisoformat(str(ts))
    except (ValueError, TypeError):
        return datetime.min


def main():
    parser = argparse.ArgumentParser(description="Milvus 知识库到期数据管理")
    parser.add_argument("--list", action="store_true", help="仅列出到期条目，不执行删除")
    parser.add_argument("--days", type=int, default=365, help="保存天数，默认 365 天(1年)")
    args = parser.parse_args()

    client = MilvusClient(
        uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
        db_name=config.MILVUS_DATABASE_NAME,
    )
    collection = config.MILVUS_COLLECTION_NAME
    if not client.has_collection(collection):
        print(f"✗ 集合不存在: {collection}")
        return

    client.load_collection(collection)

    # 1. 拉取全部条目的 id/text/timestamp（知识库规模有限，直接全量查询）
    rows = client.query(
        collection_name=collection,
        filter="",
        output_fields=["id", "text", "timestamp"],
        limit=16384,
    )
    if not rows:
        print("集合为空，无数据可管理。")
        return

    # 2. 筛选到期条目
    deadline = datetime.now() - timedelta(days=args.days)
    expired = [r for r in rows if parse_ts(r.get("timestamp")) < deadline]

    print("=" * 60)
    print(f"集合 {collection} | 总条目: {len(rows)} | 到期(>{args.days}天): {len(expired)}")
    print("=" * 60)

    for r in expired[:20]:
        preview = (r.get("text") or "")[:40].replace("\n", " ")
        print(f"  [{r.get('id')[:8]}…] {parse_ts(r.get('timestamp')).isoformat()}  {preview}…")
    if len(expired) > 20:
        print(f"  …（其余 {len(expired) - 20} 条省略）")

    if not expired:
        print("没有到期条目，无需处理。")
