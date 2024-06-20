# 环境修复记录（2026-10-15）

修复前的报错：`pymysql.err.OperationalError: (1049, "Unknown database 'subjects_kg'")`

## 修复清单

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 1 | MySQL 连不上（1049） | `subjects_kg` 库不存在 | `python scripts/init_mysql.py` 建库+建表+灌数据（467 条） |
| 2 | `bm25_search.py` 直接运行报 `ModuleNotFoundError: mysql_qa` | 该文件缺 `sys.path` 引导（同目录其它文件都有） | 已补 `sys.path.insert(项目根)` |
| 3 | Milvus 报 `database not found[itcast]` | Milvus 里的 `itcast` 数据库不存在 | 已 `create_database("itcast")` |
| 4 | `from rag_qa import ...` 直接 OSError | `new_rag_system.py:33` 硬编码 Windows 路径 `D:\workspace\...` | 改为 `os.path.join(rag_qa_path, ...)` + 离线模式 |
| 5 | `.md` 文件被静默跳过（课程大纲没进库） | `UnstructuredMarkdownLoader` 需要 NLTK 的 `punkt_tab` 等数据 | 下载到项目内 `nltk_data/` + `document_processor.py` 里自动注入搜索路径 |
| 6 | MySQL 5.7 建表可能报 1064 | 原 `create_table()` 在列定义后写 `# 注释`（8.0.29+ 才支持） | 初始化脚本把注释放在 SQL 外；列类型 `VARCHAR(1000)` → `TEXT` |

## 验证结果

- MySQL：`jpkb` 467 条（JAVA 158 / Python 309），最长答案 999 字符
