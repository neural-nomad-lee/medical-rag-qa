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
- BM25：库内原句命中（softmax=1.0000），库外问题走 RAG（≤0.387）
- 文档处理：3 个文件全部加载（docx + pdf + md），切分 130 个子块
  - 修复前只加载 2 个（md 被跳过），只有 79 个子块

## 待你执行（最后一步）

```bash
conda activate EduRAG
cd <项目根>
cd rag_qa && python main.py --data-processing --data-dir ./data
```
把 130 个子块编码进 Milvus（BGE-M3 在 CPU 上跑，视机器约 1~3 分钟）。

完成后即可：
```bash
python new_main.py        # 命令行问答
python app.py             # Web 服务 → http://127.0.0.1:8080
```
