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

