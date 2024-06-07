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

