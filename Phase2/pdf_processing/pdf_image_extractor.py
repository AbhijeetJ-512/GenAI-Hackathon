# pdf_image_extractor.py
import os
import fitz  # PyMuPDF

class PDFImageExtractor:
    def extract_images_from_pdf(self, pdf_path, pdf_name):
        pdf_document = fitz.open(pdf_path)
        images_by_page = {}
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            images = page.get_images(full=True)
            image_filenames = []
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_extension = base_image["ext"]
                image_filename = f"images/{pdf_name}_page{page_num + 1}_img{img_index + 1}.{image_extension}"
                os.makedirs(os.path.dirname(image_filename), exist_ok=True)
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)
                image_filenames.append(image_filename)
            images_by_page[page_num + 1] = image_filenames
        return images_by_page
