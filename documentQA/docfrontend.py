import streamlit as st
from docbackend import generate_embeddings, find_relevant_document, get_gpt4_answer
import fitz  

# Streamlit frontend
st.title('Document Upload and Question Answering App')

# Upload a single PDF document
uploaded_file = st.file_uploader("Choose a PDF document", type="pdf", accept_multiple_files=False)
if uploaded_file:
    # Process the uploaded PDF file
    if uploaded_file.type == "application/pdf":
        # Open PDF and extract text
        pdf_file = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        doc_text = ""
        for page in pdf_file:
            doc_text += page.get_text()
        
        # Generate embeddings for the uploaded document
        embeddings, index = generate_embeddings([doc_text])
        st.write(f'Uploaded PDF file: {uploaded_file.name}')
        st.write("Embeddings generated and stored in Faiss.")

        # User question input
        question = st.text_input('Ask a question:')
        if question:
            # Find relevant document using Faiss 
            relevant_doc = find_relevant_document(question, index, [doc_text])
            answer = get_gpt4_answer(question, relevant_doc)
            st.write(f'Answer: {answer}')

