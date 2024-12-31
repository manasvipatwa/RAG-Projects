import openai
import yfinance as yf
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests


# Load API keys from the config file
with open('config.json') as fd:
    conf = json.loads(fd.read())
    openai.api_key = conf["openai_key"]
    serp_api_key = conf["serpapi_key"]

# Create FastAPI app
app = FastAPI()

# Define request body schema
# class CompanyRequest(BaseModel):
#     company_name: str

class QueryRequest(BaseModel):
    query: str
    
def find_ticker(company_name):
    # Use OpenAI to search for the company ticker
    response = openai.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"What is the stock ticker symbol for {company_name}? Give only the ticker and no other characters.",
            }
        ],
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=16
    )
    ticker = response.choices[0].message.content
    return ticker


def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    return hist['Close'].iloc[-1]


def give_insights(company_name, price):
    # Use OpenAI to search for the company ticker
    response = openai.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"The stock price for {company_name} is {price}. Give only 3-4 lines of summary on why this {price} could be the value for {company_name}. Mention the timeline of your knowledge.",
            }
        ],
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=300
    )
    ticker = response.choices[0].message.content
    return ticker

def get_stock_info_from_serpapi(query):
    # Use SerpApi to answer the general query about stock
    search_url = f"https://serpapi.com/search?q={query}&api_key={serp_api_key}"
    response = requests.get(search_url)
    
    if response.status_code == 200:
        data = response.json()
        
        # Get the first 3 organic results
        results = data.get("organic_results", [])
        
        # If there are less than 3 results, we just get as many as available
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

def summarize_serpapi_result(query, serpapi_info):
    # Prepare the content from the first three results
    combined_content = "\n\n".join([f"Content: {item['content']}\nCitation: {item['citation']}" for item in serpapi_info])
    
    # Use OpenAI to summarize and frame the answer based on the SerpApi content
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": f"Given the query: '{query}', provide a concise, clear, and relevant response based on the following content from multiple sources:\n{combined_content}\nPlease summarize and frame the information appropriately."
            }
        ],
        temperature=0.5,
        max_tokens=200
    )
    summarized_answer = response.choices[0].message.content
    return summarized_answer


# def get_stock_info_from_serpapi(query):
#     # Use SerpApi to answer the general query about stock
#     search_url = f"https://serpapi.com/search?q={query}&api_key={serp_api_key}"
#     response = requests.get(search_url)
    
#     if response.status_code == 200:
#         data = response.json()
        
#         # Return raw response including citations (sources)
#         result = data.get("organic_results", [{}])[0]  # First organic result
        
#         # Extract the relevant content and citations
#         content = result.get("snippet", "No snippet found.")
#         citation = result.get("link", "No source link available.")
        
#         # Combine content with citation
#         return {
#             "content": content,
#             "citation": citation
#         }
#     else:
#         return {"error": "Failed to retrieve stock info from SerpApi"}

@app.post("/get_stock_price/")
async def get_stock_price_api(request: Request):
    try:
        # Correct way to get the JSON body
        body = await request.json()

        query = body.get('query')
        if not query:
            return {"error": "Query not found in the request body"}

        # Get the raw response from SerpApi based on the query
        stock_info = get_stock_info_from_serpapi(query)
        if "error" in stock_info:
            return stock_info
        
        # Summarize the SerpApi result using GPT
        summarized_answer = summarize_serpapi_result(query, stock_info)

        return {
            "query": query,
            "answer": summarized_answer,
            "citations": [item['citation'] for item in stock_info]
        }
    except Exception as e:
        # If there is any error, print it and return an error message
        print(f"Error: {str(e)}")  # Logging the error
        return {"error": f"Error: {str(e)}"}


# @app.post("/get_stock_price/")
# async def get_stock_price_api(request: Request):
#     try:
#         # Correct way to get the JSON body
#         body = await request.json()  # This is asynchronous and works correctly

#         # Logging the body to debug
#         print(f"Received request body: {body}")

#         query = body.get('query')
#         if not query:
#             return {"error": "Query not found in the request body"}

#         # Process the query as usual
#         stock_info = get_stock_info_from_serpapi(query)
#         if "error" in stock_info:
#             return stock_info
        
#         return {
#             "query": query,
#             "serpapi_info": stock_info
#         }
#     except Exception as e:
#         # If there is any error, print it and return an error message
#         print(f"Error: {str(e)}")  # Logging the error
#         return {"error": f"Error: {str(e)}"}



# @app.post("/get_stock_price/")
# async def get_stock_price_api(request: CompanyRequest):
#     # company_name = request.company_name
#     # ticker = find_ticker(company_name)
    
#     # try:
#     #     price = get_stock_price(ticker)
#     #     insights = give_insights(company_name, price)
#     #     return {"company": company_name, "ticker": ticker, "price": price, "insights": insights}
#     # except Exception as e:
#     #     return {"error": "No price data found for this company"}
    
#     query = request.query
#     try:
#         # Get the raw response from SerpApi based on the query
#         stock_info = get_stock_info_from_serpapi(query)
        
#         if "error" in stock_info:
#             return stock_info
        
#         return {
#             "query": query,
#             "serpapi_info": stock_info
#         }
#     except Exception as e:
#         return {"error": f"Error: {str(e)}"}

