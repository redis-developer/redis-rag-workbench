import os

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from PIL import Image
from gradio_pdf import PDF


def process_file(file, chunk_size: int, chunking_technique: str):
    # Load the PDF
    loader = PyPDFLoader(file.name)
    documents = loader.load()

    # Choose the appropriate text splitter based on the chunking technique
    if chunking_technique == "Semantic":
        text_splitter = SemanticChunker(
            OpenAIEmbeddings(),
        )
    else:  # Recursive Character Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=20,
            length_function=len,
        )

    # Split the documents
    split_docs = text_splitter.split_documents(documents)

    # Extract the file name
    file_name = os.path.basename(file.name)

    return split_docs, file_name


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

def setup_pdf_viewer(file_path: str, starting_page: int = 1) -> PDF:
    """Create a PDF viewer component for the given file."""
    return PDF(value=file_path, starting_page=starting_page)
