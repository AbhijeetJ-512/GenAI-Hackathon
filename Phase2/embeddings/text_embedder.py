# text_embedder.py
from sentence_transformers import SentenceTransformer
from database import batch_data

class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
    
    def process_and_encode_text(self, data, index):
        print("Started Text embedding...")
        results = []
        for key, value in data.items():
            pdf_name, page_num = key.split('_page_')
            text_chunks = value["text_chunks"]
            for chunk_number, text_chunk in enumerate(text_chunks, start=1):
                embeddings = self.embedding_model.encode(text_chunk).tolist()
                metadata = {
                    "pdf name": pdf_name,
                    "page no": page_num,
                    "chunk no": chunk_number,
                    "Text": text_chunk
                }
                result = {
                    'id': f"{pdf_name}_{page_num}_{chunk_number}",
                    'values': embeddings,
                    "metadata": metadata
                }
                results.append(result)

        batch_size = 100
        for batch in batch_data(results, batch_size=batch_size):
            index.upsert(vectors=batch)
        print("Text embeddings uploaded successfully!")
