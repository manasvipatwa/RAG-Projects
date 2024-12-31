import streamlit as st
import requests

# Streamlit title
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


