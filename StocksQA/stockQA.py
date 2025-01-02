import openai
import yfinance as yf
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import opik
from opik.integrations.openai import track_openai
from opik import track, opik_context
import ollama
import streamlit as st
import os

# Ollama API URL
ollama_url = "http://localhost:11434/api/generate"  # Default Ollama API URL

serp_api_key = os.getenv("SERP_API_KEY")


# Create FastAPI app
app = FastAPI()

# Define request body schema
class QueryRequest(BaseModel):
    query: str

# FastAPI Route for Stock Query
@app.post("/get_stock_price/")
async def get_stock_price_api(request: Request):
    try:
        body = await request.json()
        query = body.get('query')
        
        if query:
            # Get the raw response from SerpApi based on the query
            stock_info = get_stock_info_from_serpapi(query)
            if "error" in stock_info:
                return stock_info
            
            # Summarize the SerpApi result using Ollama's Llama model
            summarized_answer = summarize_serpapi_result(query, stock_info)

            return {
                "query": query,
                "answer": summarized_answer,  # The summary should be the final output
                "citations": [item['citation'] for item in stock_info]
            }
        else:
            return {"error": "No query provided."}
    
    except Exception as e:
        return {"error": str(e)}

# Function to get stock information from SerpApi
def get_stock_info_from_serpapi(query):
    # Use SerpApi to answer the general query about stock
    search_url = f"https://serpapi.com/search?q={query}&api_key={serp_api_key}"
    response = requests.get(search_url)
    
    if response.status_code == 200:
        data = response.json()
        
        # Get the first 3 organic results
        results = data.get("organic_results", [])
        top_results = results[:3]

        # Extracting content and citations from the top 3 results
        result_data = []
        for result in top_results:
            content = result.get("snippet", "No snippet found.")
            citation = result.get("link", "No source link available.")
            result_data.append({
                "content": content,
                "citation": citation
            })
        
        return result_data
    else:
        return {"error": "Failed to retrieve stock info from SerpApi"}

# Function to summarize results using Ollama's Llama model
@track(tags=['ollama', 'python-library', 'summarization'])
def summarize_serpapi_result(query, serpapi_info):
    # Prepare the content from the first three results
    combined_content = "\n\n".join([f"Content: {item['content']}\nCitation: {item['citation']}" for item in serpapi_info])
    
    # Create the prompt for the model
    prompt = f"Given the query: '{query}', provide a concise, clear, and relevant response based on the following content from multiple sources:\n{combined_content}\nPlease summarize and frame the information appropriately."

    # Send the request to the Ollama API for summarization
    response = requests.post(ollama_url, json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False  
    })

    response_data = response.json()

    # Track the response data with Opik
    if 'response' in response_data:
        # Extract the response and other metrics from Ollama's response
        summarized_answer = response_data['response'].strip()  

        # Update Opik context with relevant metadata
        opik_context.update_current_span(
            metadata={
                'model': response_data['model'],
                'eval_duration': response_data['eval_duration'],
                'load_duration': response_data['load_duration'],
                'prompt_eval_duration': response_data['prompt_eval_duration'],
                'prompt_eval_count': response_data['prompt_eval_count'],
                'done': response_data['done'],
                'done_reason': response_data['done_reason'],
            },
            usage={
                'completion_tokens': response_data['eval_count'],
                'prompt_tokens': response_data['prompt_eval_count'],
                'total_tokens': response_data['eval_count'] + response_data['prompt_eval_count']
            }
        )

        return summarized_answer
    else:
        raise ValueError("Error fetching summary from Ollama API")

# Streamlit Frontend for User Input
st.title("Stock Query Answering System")

# Input for general stock-related question
st.subheader("Ask a stock-related question:")
query = st.text_input("Enter query:")

# If a question is entered
if query.strip():  
    try:
        response = requests.post("http://localhost:8000/get_stock_price/", json={"query": query})
        response.raise_for_status()  

        # If the request was successful
        if response.status_code == 200:
            data = response.json()
            if "answer" in data:
                # st.write(f"Query: {data['query']}")
                st.subheader("Response:")
                st.write(f"Answer: {data['answer']}")

                # Display citations
                st.subheader("References:")
                for i, citation in enumerate(data['citations'], 1):
                    st.write(f"[{i}] {citation}")
                    
            else:
                st.write(f"Error: {data.get('error', 'An error occurred.')}")
        else:
            st.write("Failed to retrieve data. Please try again.")
    except requests.exceptions.RequestException as e:
        st.write(f"Request error: {e}")
else:
    st.write("Please enter a valid query.")
