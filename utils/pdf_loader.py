from langchain_community.document_loaders import PyPDFLoader
from typing_extensions import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(uploaded_file_path: str) -> List:
    loader = PyPDFLoader(uploaded_file_path)
    documents = loader.load()
    return documents


def divide_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    return texts