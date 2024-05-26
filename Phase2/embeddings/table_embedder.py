# table_embedder.py
import pandas as pd
from tqdm.auto import tqdm
from database import batch_data
from sentence_transformers import SentenceTransformer

class TableEmbedder:
    def __init__(self, model_name="deepset/all-mpnet-base-v2-table"):
        self.retriever = SentenceTransformer(model_name)
    
    def _preprocess_tables(self, tables):
        processed = []
        table = []
        for table in tables:
            df = pd.DataFrame(table)
            processed_table = df.to_csv(index=False, header=False)
            processed.append(processed_table)
            table.append(df)
        return table, processed

    def process_and_encode_tables(self, data, index):
        print("Started Table embedding...")
        tables = []
        for key, value in data.items():
            y = value['tables']
            p, q = self._preprocess_tables(y)
            tables.extend(q)

        batch_size = 64
        for i in tqdm(range(0, len(tables), batch_size)):
            i_end = min(i + batch_size, len(tables))
            batch = tables[i:i_end]
            emb = self.retriever.encode(batch).tolist()
            ids = [f"{idx}" for idx in range(i, i_end)]
            to_upsert = list(zip(ids, emb))
            _ = index.upsert(vectors=to_upsert)
        print("Table embeddings uploaded successfully!")
