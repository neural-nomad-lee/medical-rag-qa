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
