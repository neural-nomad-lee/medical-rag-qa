#!/usr/bin/env python3
# coding: utf-8
"""
init_mysql.py —— 一键初始化知识库（建库 → 建表 → 导 CSV → 校验），幂等可重复执行

【为什么需要这个脚本】
  MySQLClient 连的是"库"（database=med_kg），而库必须先存在。
  报错 pymysql.err.OperationalError: (1049, "Unknown database 'med_kg'")
  就是"库还没建"——不是代码 bug，是初始化步骤没做。

【为什么不用 mysql_client.create_table() 和 insert_data()】
  1. 原 create_table() 把 `# 注释` 写在了 CREATE TABLE 的列定义后面。
     实测本机 MySQL 版本为 5.7.28，**不支持**这种写法（8.0.29+ 才部分支持），
     直接执行会抛 1064 语法错。所以本脚本把注释全部放在 SQL 外面。
  2. 原列类型是 VARCHAR(1000)，而 CSV 里答案最长 999 字符（贴着上限）。
     一旦以后补充更长的答案就会被截断/报错。本脚本改用 TEXT。
  3. 原 insert_data() 没有幂等保护，重复执行会让数据翻倍。
     本脚本会先判重，已导入就跳过。

【用法】
  python scripts/init_mysql.py
"""

import os
import sys

import pymysql

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "scripts" else HERE
sys.path.insert(0, ROOT)

# 复用项目自己的配置类（保证与业务代码读的是同一份配置）
from base.config import config

# 兼容性处理：手册里的"改进版 Config"新增了 MYSQL_CHARSET 字段，
# 而项目当前的 base/config.py 是原版、没有该字段。
# 用 getattr 兜底，保证脚本在两种配置下都能跑。
MYSQL_CHARSET = getattr(config, "MYSQL_CHARSET", "utf8mb4")


def connect(use_db=True):
    """
    建立连接。
    :param use_db: True → 连到 med_kg 库；False → 只连服务器（建库时需要）
    注意 charset='utf8mb4'：本机 MySQL 5.7 服务端默认是 latin1，
         不显式指定会导致中文写入报 Incorrect string value 或乱码。
    """
    kwargs = dict(
        host=config.MYSQL_HOST,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        charset=MYSQL_CHARSET,
        autocommit=False,
    )
    if use_db:
        kwargs["database"] = config.MYSQL_DATABASE
    return pymysql.connect(**kwargs)


def step1_create_database():
    """第 1 步：建库（utf8mb4，支持中文与 emoji）"""
    print(f"[1/4] 建库 {config.MYSQL_DATABASE}（若不存在）")
    conn = connect(use_db=False)
    try:
        with conn.cursor() as cur:
            # IF NOT EXISTS 保证幂等；utf8mb4_general_ci 是 5.7/8.0 通吃的排序规则
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{config.MYSQL_DATABASE}` "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"
            )
        conn.commit()
        print("      ✓ 库已就绪")
    finally:
        conn.close()


def step2_create_table():
    """第 2 步：建 ylkb 医疗知识库表（注意：注释全部写在 SQL 外部！）"""
