"""
OCR介绍:
    简单说就是'光学字符识别'技术, Optical Character Recognition, 能让电脑'看懂'图片里的文字,
    把图片上的手写体, 印刷体文字转换成可编辑的文本 -> 例如: 照片里的笔记,扫描版pdf的文字等.
"""




import cv2
import fitz  # TODO pyMuPDF里面的fitz包，不要与pip install fitz混淆
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Iterator
from rag_qa.med_document_loaders.med_ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import CharacterTextSplitter
# PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR。
# 这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.6, 0.6)


class OCRPDFLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        line = self.pdf2text()
        yield Document(page_content=line, metadata={"source": self.file_path})

    def pdf2text(self):
        ocr = get_ocr()
        # 打开pdf文件
        doc = fitz.open(self.file_path)
        ## 获取页数
        # print(f'len(doc)-->{len(doc)}')
        resp = ""
        b_unit = tqdm(total=doc.page_count, desc="OCRPDFLoader context page index: 0")
        for i, page in enumerate(doc):
            b_unit.set_description("OCRPDFLoader context page index: {}".format(i))
            b_unit.refresh()
            # 提取文本：默认使用 "text" 模式提取文本。
            text = page.get_text("text")
            resp += text + "\n"
            # print(f'resp-->{resp}')
            # 获取图片：获得所有显示的图像的元信息列表。
            # 它适用于所有文档类型，不仅限于 PDF。
            img_list = page.get_image_info(xrefs=True)
            # print(f'img_list--》{img_list}')
            # print(f'img_list--》{len(img_list)}')
