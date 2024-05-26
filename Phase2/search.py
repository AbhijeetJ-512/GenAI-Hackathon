import streamlit as st
import random
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer
import spacy
import os
# Load BART model and tokenizer for summarization
bart_summarizer = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
nlp = spacy.load('en_core_web_sm')

def answer_question(text):
    # Tokenize input text for BART
    bart_inputs = bart_tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    # Generate summary
    summary_ids = bart_summarizer.generate(bart_inputs['input_ids'], max_length=200, num_beams=5, early_stopping=True)
    # Decode the generated summary
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def perform_similarity_search(query_text, index_text, index_table):
    model_text = SentenceTransformer("all-MiniLM-L6-v2")
    model_table = SentenceTransformer("deepset/all-mpnet-base-v2-table")
    
    # Get embeddings for the query text
    query_text_embedding = model_text.encode([query_text])[0]
    query_table_embedding = model_table.encode([query_text])[0]
    
    # Perform the similarity search for text and table embeddings
    results_text = index_text.query(
        vector=query_text_embedding.tolist(),
        top_k=2,
        include_values=True,
        include_metadata=True
    )
    
    results_table = index_table.query(
        vector=query_table_embedding.tolist(),
        top_k=2,
        include_values=True,
        include_metadata=True
    )
    # Combine results from both searches
    combined_results = results_text["matches"] + results_table["matches"]
    
    # Sort combined results by score in descending order
    sorted_results = sorted(combined_results, key=lambda x: x['score'], reverse=True)
    score=0
    # Extract context text from the top result
    if sorted_results and 'metadata' in sorted_results[0] and 'Text' in sorted_results[0]['metadata']:
        top_result = sorted_results[0]
        context_text = top_result['metadata']['Text']
        result_id = top_result['id']
        score = top_result['score']
        pdf_name = top_result['metadata']['pdf name']
        page_no = top_result['metadata']['page no']
        chunk_no = int(top_result['metadata']['chunk no'])
    else:
        context_text = ""
        result_id = None
    
    # Print the id of the top result
    print(f"Top result ID: {result_id} {score} {pdf_name} {page_no} {chunk_no}")
    answer=None
    if score < 0.5:
        nouns_and_pronouns = []
        doc = nlp(query_text)
        # print(doc)
        for token in doc:
            # print(token.pos_)
            if token.pos_ in ['NOUN', 'PRON','PROPN','SCONJ','VERB']:
                nouns_and_pronouns.append(token.text)
        # print("Nouns and Pronouns:", nouns_and_pronouns)
        answer=f"Can you please provide more details about {', '.join(nouns_and_pronouns)}?"
        print(answer)
    else:
        answer = answer_question(context_text)
        print(answer)
    response = f"Chatty: {answer}"
    image_path = f"images/{pdf_name}_page{page_no}_img{chunk_no}"

# Check if the image file exists with different extensions
    image_extensions = ['jpg', 'jpeg', 'png']
    image_file = None
    for ext in image_extensions:
        if os.path.exists(f"{image_path}.{ext}"):
            image_file = f"{image_path}.{ext}"
            break   
    
    print(image_file)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        if image_file:
            image = open(image_file,'rb').read()
            st.image(image, caption='Image', use_column_width=True)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

