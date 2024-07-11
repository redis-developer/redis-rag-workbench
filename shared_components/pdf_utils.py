import re
import fitz
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from typing import Tuple, List
from langchain.schema import Document


def process_file(file: str) -> Tuple[List[Document], str]:
    loader = PyPDFLoader(file.name)
    documents = loader.load()
    pattern = r"/([^/]+)$"
    match = re.search(pattern, file.name)
    file_name = match.group(1)
    return documents, file_name


def render_file(file: str, page_num: int = 0) -> Image.Image:
    doc = fitz.open(file.name)
    try:
        page = doc[page_num]
    except IndexError:
        print(f"Invalid page number: {page_num}, defaulting to page 0")
        page = doc[0]

    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image


def render_first_page(file) -> Image.Image:
    doc = fitz.open(file.name)
    page = doc[0]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    return image
