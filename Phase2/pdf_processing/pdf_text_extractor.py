# pdf_text_extractor.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

class PDFTextExtractor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def extract_text_from_pdf(self, pdf_path):
        text_by_page = {}
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                text_chunks = self.text_splitter.split_text(text)
                text_by_page[page_num + 1] = text_chunks
        return text_by_page
