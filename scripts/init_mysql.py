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
