# Vector Database for Semantic Search

This repository provides code for setting up a vector database optimized for semantic search using Sentence Transformers and Pinecone. Semantic search allows finding documents or text passages similar in meaning to a query rather than just keyword matching.

## Setup

1. **Create a Virtual Environment:**
```bash
    conda create --name myenv python=3.9
    conda activate myenv
```
2. **Install Dependencies:**
```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
```
3. **API Key for Pinecone:**

Obtain an API key from Pinecone and add it to the main file.

4. **Configure PDF File Paths:**

Edit the main.py file to provide the paths to the PDF files. Ensure that the file paths are correctly formatted and located in the specified location within the code.
5. **Run the code**
```bash
    streamlit run main.py
```
This command will start the Streamlit application, allowing you to interact with the semantic search functionality.
