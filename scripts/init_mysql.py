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
    print("[2/4] 建表 ylkb（与 mysql_qa/db/mysql_client.py 查询的表名保持一致）")
    conn = connect()
    try:
        with conn.cursor() as cur:
            # 字段说明：
            #   department 科室名称（对应 医疗导诊FAQ.csv 的"科室"列，VARCHAR(20) 足够）
            #   question     问题原文
            #   answer       答案原文 → 用 TEXT 更安全，避免长答案被截断
            #   expire_at    到期时间：NULL=永久保留；RAG 沉淀的答案设为半年后到期，
            #                由应用层定期清理（见 app.py 的 cleanup_expired_knowledge）
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ylkb (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    department VARCHAR(20),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    expire_at DATETIME NULL,
                    INDEX idx_expire_at (expire_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        conn.commit()
        print("      ✓ 表已就绪")
    finally:
        conn.close()


def step3_import_csv():
    """第 3 步：导入医疗导诊FAQ.csv（判重，已导入则跳过）"""
    csv_path = os.path.join(ROOT, "mysql_qa", "data", "医疗导诊FAQ.csv")
    print(f"[3/4] 导入 CSV：{os.path.relpath(csv_path, ROOT)}")

    if not os.path.exists(csv_path):
        print(f"      ✗ 找不到 CSV：{csv_path}")
        return 0

    import pandas as pd

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ylkb")
            existing = cur.fetchone()[0]
            if existing > 0:
                print(f"      已有 {existing} 条数据，跳过导入（如需重灌请先清空表）")
                return existing

            # pandas 能正确处理"答案字段里含换行符"的 CSV（普通逐行读会把一条记录拆成多行）
            # FAQ 精准问答数据为人工维护的种子知识，expire_at 置 NULL 表示永久保留。
            data = pd.read_csv(csv_path)
            rows = [
                (str(r["科室"]), str(r["问题"]), str(r["答案"]), None)
                for _, r in data.iterrows()
            ]
            # executemany：比逐条 execute 快一个数量级
            cur.executemany(
                "INSERT INTO ylkb(department, question, answer, expire_at) VALUES(%s, %s, %s, %s)",
                rows,
            )
        conn.commit()
        print(f"      ✓ 导入完成：{len(rows)} 条")
        return len(rows)
    except Exception as e:
        conn.rollback()
        print(f"      ✗ 导入失败，已回滚：{type(e).__name__}: {e}")
        raise
    finally:
        conn.close()


def step4_verify():
    """第 4 步：校验（条数 / 科室分布 / 最长答案长度）"""
    print("[4/4] 校验")
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ylkb")
            total = cur.fetchone()[0]
            print(f"      总条数：{total}")

            cur.execute("SELECT department, COUNT(*) FROM ylkb GROUP BY department")
            for name, cnt in cur.fetchall():
                print(f"      科室分布：{name} → {cnt} 条")

            cur.execute("SELECT MAX(CHAR_LENGTH(answer)), MAX(CHAR_LENGTH(question)) FROM ylkb")
            max_ans, max_q = cur.fetchone()
            print(f"      最长答案 {max_ans} 字符 ｜ 最长问题 {max_q} 字符")

            # 抽查一条，确认中文没乱码
            cur.execute("SELECT question, LEFT(answer, 40) FROM ylkb LIMIT 1")
            q, a = cur.fetchone()
            print(f"      抽样：Q={q[:34]} … A={a}…")

            # 验证 SQL 注入防护占位符可用（顺带验证查询路径）
            cur.execute("SELECT answer FROM ylkb WHERE question=%s", ("血压低挂什么科？",))
            row = cur.fetchone()
            print(f"      fetch_answer 通道：{'命中' if row else '未命中（该问题可能不在本数据集中）'}")

            # 验证会话历史表能否创建（这是 new_main.py 启动时会执行的 DDL）
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            conn.commit()
            print("      ✓ conversations 表已就绪（多轮对话历史）")
        return total
    finally:
        conn.close()


def main():
    print("=" * 60)
    print("知识库初始化（幂等，可重复执行）")
    print(f"MySQL: {config.MYSQL_HOST} ｜ 库: {config.MYSQL_DATABASE} ｜ 字符集: {MYSQL_CHARSET}")
    print("=" * 60)
    step1_create_database()
    step2_create_table()
    step3_import_csv()
    total = step4_verify()
    print("=" * 60)
    if total:
        print(f"✅ 完成：ylkb 共 {total} 条，可以运行 python mysql_qa/db/mysql_client.py 验证")
    else:
        print("⚠️ 表为空，请检查 CSV 路径与内容")


if __name__ == "__main__":
    main()
