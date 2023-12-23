from typing import Iterator
from rag_qa.med_document_loaders.med_ocr import get_ocr
# 导入必要的模块
from tqdm import tqdm
from docx.table import _Cell, Table  # 用于处理表格
from docx.oxml.table import CT_Tbl  # 用于处理表格XML结构
from docx.oxml.text.paragraph import CT_P  # 用于处理段落XML结构
from docx.text.paragraph import Paragraph  # 用于处理段落内容
from docx import Document as Docu1
from docx.document import Document as Docu2
from docx import ImagePart  # 用于处理Word文档和图片
from PIL import Image  # 用于处理图片
from io import BytesIO  # 用于将字节流转换为图片
import numpy as np  # 用于处理数组
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class OCRDOCLoader(BaseLoader):
    """An example document loader that reads a file line by line."""
