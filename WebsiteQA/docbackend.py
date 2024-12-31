import openai
import numpy as np
from faiss import IndexFlatL2
from transformers import AutoTokenizer, AutoModel
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import json
# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# Load API keys from the config file
with open('config.json') as fd:
    conf = json.loads(fd.read())
    openai.api_key = conf["openai_key"]

# Function to generate embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Function to generate embeddings and store them in Faiss
def generate_embeddings(documents):
    embeddings = get_embeddings(documents)
    
    # Store embeddings in Faiss
    index = IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return embeddings, index

# Function to find the most relevant document based on Faiss similarity search
def find_relevant_document(question, faiss_index, documents):
    question_embedding = get_embeddings([question])
    D, I = faiss_index.search(question_embedding, k=1)  
    return documents[I[0][0]]

# Function to query GPT-4 for an answer based on the relevant document
def get_gpt4_answer(question, document):
    response = openai.chat.completions.create(
        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Document: {document}\n\nQuestion: {question}\nAnswer:"}
    ],
        model="gpt-4",
        temperature=0,
        max_tokens=200
    )
    return response.choices[0].message.content