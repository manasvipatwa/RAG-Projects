import streamlit as st
import json
from scrapegraphai.graphs import SmartScraperGraph
from firecrawl import FirecrawlApp
from docbackend import generate_embeddings, find_relevant_document, get_gpt4_answer
from pydantic import BaseModel

# Streamlit frontend
st.title('Website Scraping and Q&A System')

# Ask for the URL input
url_input = st.text_input("Enter the URL of the website to scrape:")

# Check if URL is provided and start scraping
if url_input:
    documents = []
    # Load API key from config.json
    with open('config.json') as fd:
        conf = json.loads(fd.read())
        firecrawl_api_key = conf["firecrawl_api_key"]  

    # Initialize FireCrawlApp instance
    app = FirecrawlApp(api_key=firecrawl_api_key)

    try:
        # Scrape data using FireCrawl
        scrape_result = app.scrape_url(url_input, params={'formats': ['markdown', 'html']})
        
        # Extract content from the markdown field
        if 'markdown' in scrape_result:
            scraped_text = scrape_result['markdown']  

            if scraped_text:
                st.write(f"Scraped data from {url_input}")  

                documents.append(scraped_text) 
            else:
                st.write(f"No valid content extracted for {url_input}. Please check the URL.")
        else:
            st.write(f"Error: 'markdown' key not found in the FireCrawl response. Full response shown above.")

    except Exception as e:
        st.error(f"Error while scraping: {e}")

    # Generate embeddings for the scraped content
    if documents:
        embeddings, index = generate_embeddings(documents)
        st.write("Embeddings for scraped data generated and stored.")

        # Ask the user for a question about the scraped content
        question = st.text_input('Ask a question:')

        if question:
            # Generate relevant insights using GPT-4
            relevant_doc = find_relevant_document(question, index, documents)

            # Query GPT-4 for insights
            answer = get_gpt4_answer(question, relevant_doc)
            st.write(f'Answer: {answer}')
    else:
        st.write("No valid scraped content to generate embeddings.")
