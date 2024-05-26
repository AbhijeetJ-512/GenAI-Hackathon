# main.py
import torch
import clip
import os
import streamlit as st
from database import setup_database
from utils.file_utils import save_json, load_json
from utils.data_utils import convert_keys_to_strings
from pdf_processing.pdf_text_extractor import PDFTextExtractor
from pdf_processing.pdf_image_extractor import PDFImageExtractor
from pdf_processing.pdf_table_extractor import PDFTableExtractor
from embeddings.text_embedder import TextEmbedder
from embeddings.image_embedder import ImageEmbedder
from embeddings.table_embedder import TableEmbedder
from search import perform_similarity_search
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    pdf_directory = '../Pdfs/' #problem with punch.pdf 
    output_file = 'pdf_content_bert.json'
    pinecone_api_key = "a6b671ce-deea-4868-9a3b-eff77d0d84f5"
    
    # Initialize models
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder(clip_model)
    table_embedder = TableEmbedder()

    # Initialize extractors
    text_extractor = PDFTextExtractor()
    image_extractor = PDFImageExtractor()
    table_extractor = PDFTableExtractor()

    # Process PDF files
    pdf_content = {}
    for root, dirs, files in os.walk(pdf_directory):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                pdf_name = os.path.basename(pdf_path).split('.')[0]

                text_by_page = text_extractor.extract_text_from_pdf(pdf_path)
                images_by_page = image_extractor.extract_images_from_pdf(pdf_path, pdf_name)
                tables_by_page = table_extractor.extract_tables_from_pdf(pdf_path)

                for page_num in range(1, max(len(text_by_page), len(images_by_page), len(tables_by_page)) + 1):
                    pdf_content[(pdf_name, page_num)] = {
                        'text_chunks': text_by_page.get(page_num, []),
                        'tables': tables_by_page.get(page_num, []),
                        'images': images_by_page.get(page_num, [])
                    }
    
    content_with_str_keys = convert_keys_to_strings(pdf_content)
    save_json(content_with_str_keys, output_file)

    # Setup Pinecone index
    # api_key, index_name, dimension, metric, embedding_model
    text_index ,textdb= setup_database(pinecone_api_key, index_name="text", dimension=384)
    image_index ,imagedb= setup_database(pinecone_api_key, index_name="image", dimension=512)
    table_index, tabledb = setup_database(pinecone_api_key, index_name="table", dimension=768)
    print(textdb,tabledb,imagedb)
    # Process and upload embeddings
    if textdb:
        text_embedder.process_and_encode_text(content_with_str_keys, text_index)
    if imagedb:
        image_embeddings = image_embedder.process_and_encode_images(content_with_str_keys, image_index)
    if tabledb:
        table_embedder.process_and_encode_tables(content_with_str_keys, table_index)

    st.title("Chatty")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})


        query_text = prompt
        # query_text = input("Enter query")
        perform_similarity_search(query_text, text_index,table_index)

if __name__ == "__main__":
    main()
