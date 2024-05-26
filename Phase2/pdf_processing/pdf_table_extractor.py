# pdf_table_extractor.py
import pdfplumber

class PDFTableExtractor:
    def extract_tables_from_pdf(self, pdf_path):
        tables_by_page = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                extracted_tables = page.extract_tables()
                tables_by_page[page_num + 1] = extracted_tables
        return tables_by_page
