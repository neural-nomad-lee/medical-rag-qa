# 该脚本用于: 处理输入文档, 分块并准备向量存储.

# 导包
import os       # 文件操作

# ============================================================
# 【环境兜底】让 UnstructuredMarkdownLoader 能找到 NLTK 数据
# ------------------------------------------------------------
# 现象：加载 .md 文件时抛 LookupError: Resource punkt_tab not found
# 原因：langchain 的 UnstructuredMarkdownLoader 内部依赖 NLTK 分词/词性标注数据，
#       而 NLTK 默认只在 ~/nltk_data 等系统路径查找。
# 影响：.md 文件会被 except 分支静默跳过，导致"科室导诊指南"这类关键资料没进向量库，
#       而日志里只留一行 error，表现为"RAG 答不出大纲内容"却查不出原因。
# 解法：把项目内的 nltk_data 目录显式加入 NLTK 搜索路径（数据随项目走，换机器无需重下）。
# ============================================================
# 注意：本文件位于 rag_qa/core/，要 dirname 三次才到项目根：
#   __file__ → .../rag_qa/core/document_processor.py
#   第 1 次 → .../rag_qa/core  ｜ 第 2 次 → .../rag_qa ｜ 第 3 次 → .../integrated_qa_system（项目根）
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_NLTK_DIR = os.path.join(_PROJECT_ROOT, "nltk_data")
if os.path.isdir(_NLTK_DIR):
    os.environ.setdefault("NLTK_DATA", _NLTK_DIR)   # 仅在用户未设置时生效，不覆盖用户配置
    try:
        import nltk
        if _NLTK_DIR not in nltk.data.path:
            nltk.data.path.insert(0, _NLTK_DIR)      # 双保险：直接注入 nltk 搜索路径
    except ImportError:
        pass

from langchain_community.document_loaders import TextLoader     # 加载纯文本文件(.txt)用的工具
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader    # 加载(.md)文件的工具, 保留标题,列表等结构化信息
from langchain_text_splitters import MarkdownTextSplitter       # 针对 .md文档的文本分割器.
from datetime import datetime               # 用于记录文档加载/处理时间.
from rag_qa.med_text_spliter import ChineseRecursiveTextSplitter        # 适配中文医疗文本层级结构
from rag_qa.med_text_spliter import AliTextSplitter                     # 阿里中文优化(医疗场景专用文本分割器)
from rag_qa.med_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader  # 医疗场景专用文档加载器(支持OCR, 处理扫描版病历、检查报告、医学影像文字等.)
from base.config import Config      # 配置类
from base.logger import logger      # 日志类

# todo 1. 定义配置类 -> 获取配置参数.
# 1. 创建配置实例.
conf = Config()
# 2. 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}

# todo 2. 核心加载函数模块 -> 从指定目录递归加载所有支持类型文件, 添加原数据并返回文档列表.
def load_documents_from_directory(directory_path):
    """
    从指定目录(含子目录)加载所有支持类型的文件, 为每个文档添加原数据
    :param directory_path: 目标文件夹的绝对路径.
    :return: 加载完成的文档列表, 每个元素为 LangChain Document对象, 含: page_content 和 metadata
    """
    # 1. 初始化空列表 -> 用于存储所有加载成功的文档.
    documents = []
    # 2. 获取支持的文件扩展名结合
    supported_extensions = document_loaders.keys()
    # 3. 从目录名提取 '科室类别'原数据, 例如: '心血管内科_data' -> 提取'心血管内科'作为source
    source = os.path.basename(directory_path).replace('_data', '')
    # 4. 递归遍历目标目录及其所有子目录,  注意: os.walk()返回 当前目录路径, 子目录列表, 文件列表.
    for root, _, files in os.walk(directory_path):
        # 5. 遍历当前目录下的每个文件, 逐一处理.
        for file in files:
            # 5.1 构造文件的绝对路径  root + 文件名, 避免相对路径问题.
            file_path = os.path.join(root, file)
            # print(f'file_path: {file_path}')        # 例如: ../data/心血管内科_data\心血管内科常见病科普.md

            # 5.2 提取文件扩展名并转为小写 -> 统一格式.
            file_extension = os.path.splitext(file)[1].lower()
            # print(f'file_extension: {file_extension}')

            # 5.3 若文件类型在支持列表中, 执行加载逻辑.
            if file_extension in supported_extensions:
                try:
                    # 5.3.1 根据文件扩展名获取对应的加载器 -> 创建加载器实例.
                    loader_class = document_loaders[file_extension]
                    # 5.3.2 实例化加载器.
                    if file_extension == '.txt':
                        loader = loader_class(file_path, encoding='utf-8')  # txt文件强制使用utf-8编码
                    else:
                        loader = loader_class(file_path)
                    # 5.3.3 调用加载器的load()方法, 加载文件内容 -> 返回文档列表, 单个文件可能拆分成多个Document.
                    loaded_docs = loader.load()
                    # print(f'加载文件: {loaded_docs}')
                    # 5.3.4 给每个加载的文档添加元数据 -> 便于后续RAG检索时过滤/溯源.
                    for doc in loaded_docs:
                        # 为文档添加科室类别元数据, 例如: 心血管内科
                        doc.metadata["source"] = source
                        # 为文档添加文件路径元数据
                        doc.metadata["file_path"] = file_path
                        # 为文档添加当前时间戳元数据, 精确到秒
                        doc.metadata["timestamp"] = datetime.now().isoformat()

                    # 5.3.5 将添加元数据后的文档列表添加到总列表中.
                    documents.extend(loaded_docs)
                    # 5.3.6 记录INFO日志
                    logger.info(f'成功加载文件: {file_path}')
                except Exception as e:
                    # 5.3.7 捕获异常 -> 输出异常信息, 继续处理下一个文件.
                    logger.error(f'加载文件 {file_path} 失败: {str(e)}')
            else:
                # 5.4 若文件不再支持列表, 记录警告日志.
                logger.warning(f'不支持的文件类型: {file_path}')

    # 6. 返回加载完成的文档列表.
    return documents


# todo 3.文档处理核心函数 -> 加载文档后进行分层切分(父块 + 子块): 返回带元数据的子块列表.
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                     child_chunk_size=conf.CHILD_CHUNK_SIZE,
                     chunk_overlap=conf.CHUNK_OVERLAP):
    """
    加载文档并进行分层气氛: 先切成大粒度父块, 再将父块切成小粒度子块, 子块关联父块元数据.
    :param directory_path:  文档目录路径
    :param parent_chunk_size: 父块切分长度(大粒度, 例如: 1200)
    :param child_chunk_size: 子块切分长度(小粒度, 例如: 300)
    :param chunk_overlap: 子块重叠长度, 确保上下文关联, 例如: 50
    :return:
        child_chunks: 切分后的子块列表(每个子块含父块关联信息)
    """

    # 1. 调用加载函数 -> 从指定目录加载所有文档
    documents = load_documents_from_directory(directory_path)
    # 记录加载的文档总数日志
    logger.info(f"加载的文档数量: {len(documents)}")

    # 2. 初始化父块和子块分词器（通用）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化 Markdown 专用分词器
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 3. 初始化子块列表, 存储最终切分结果.
    child_chunks = []

    # 4. 遍历每个原始文档 -> 带索引i, 用于生成唯一ID
    for i, doc in enumerate(documents):
        # 4.1 获取文件扩展名 -> 用于判断是否是MarkDown文件, 选择对应的切分器.
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()
        is_markdown = (file_extension == '.md')     # 标记是否是MarkDown文件

        # 4.2 根据文件类型选择切分器.
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter       # 父块切分器
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter          # 子块切分器
        logger.info(f'处理文档: {doc.metadata["file_path"]}, 使用切分器: {"Markdown" if is_markdown else "ChineseRecursive"}')

        # 4.3 将文档切分成父块 -> 大粒度, 保留较多上下文.
        # parent_docs的格式: [父块1Document对象, 父块2Document对象, ...]
        parent_docs = parent_splitter_to_use.split_documents([doc])     # 传入列表格式, split_documents()函数要的是可迭代类型.

        # 5. 遍历每个父块 -> 父块切分成子块 -> 小粒度, 并于精准匹配.
        for j, parent_doc in enumerate(parent_docs):
            # 5.1 生成父块唯一id, 格式: doc_文档索引_parent_父块索引
            parent_id = f"doc_{i}_parent_{j}"
            # 5.2 给父块添加元数据 -> 唯一的ID 和 自身内容, 供子块关联.
            parent_doc.metadata["parent_id"] = parent_id
