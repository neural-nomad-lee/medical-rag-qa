from typing import Iterator
from rag_qa.med_document_loaders.med_ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from pptx import Presentation
from PIL import Image
import numpy as np
from io import BytesIO
from tqdm import tqdm


class OCRPPTLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, filepath: str) -> None:
        """Initialize the loader with a file path.

        Args:
            filepath: The path to the ppt to load.
        """
        self.filepath = filepath

    def lazy_load(self) -> Iterator[Document]:
        # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        line = self.ppt2text(self.filepath)
        yield Document(page_content=line, metadata={"source": self.filepath})

    def ppt2text(self, filepath):
        # 打开指定路径的 PowerPoint 文件
        prs = Presentation(filepath)
        print(f'prs-->{prs}')
        # 获取 OCR 功能的实例
        ocr = get_ocr()
        # 初始化一个空字符串，用于存储提取的文本内容
        resp = ""

        def extract_text(shape):
            # nonlocal指明resp非全局非局部，而是外部嵌套函数中的变量，
            # 允许内部函数访问和修改外部函数中定义的变量resp
            nonlocal resp

            # 检查形状是否有文本框
            if shape.has_text_frame:
                # 将文本框中的文本添加到resp中，并去掉前后空格
                resp += shape.text.strip() + "\n"

            # 检查形状是否为表格
            if shape.has_table:
                # 遍历表格的每一行
                for row in shape.table.rows:
                    # 遍历每一行中的每个单元格
                    for cell in row.cells:
                        # 遍历单元格中的每个段落
                        for paragraph in cell.text_frame.paragraphs:
                            # 将单元格中的文本添加到resp中，并去掉前后空格
                            resp += paragraph.text.strip() + "\n"

            # 检查形状是否为图片（shape_type == 13）
