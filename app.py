# Run Streamlit by entering streamlit run app.py into Terminal
import streamlit as st
from sentence_transformers import SentenceTransformer
import lancedb
import pandas as pd
import os
from dotenv import load_dotenv
import requests

# --- Load environment variables ---
load_dotenv()
LANCEDB_API_KEY = os.getenv("LANCEDB_API_KEY")

# --- Connect to LanceDB ---
db = lancedb.connect(
    uri="db://learning-project-dwwvuw",
    api_key=LANCEDB_API_KEY,
    region="us-east-1"
)
table = db.open_table("coding_journal")

# --- Load embedding model ---
model = SentenceTransformer("intfloat/e5-small-v2")

# --- Set up local LLM API settings ---
LOCAL_LLM_ENDPOINT = "http://127.0.0.1:1234/v1/chat/completions"
LOCAL_LLM_MODEL = "google/gemma-3-12b"

def call_local_llm(context, question):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that explains code from a learning journal."},
            {"role": "user", "content": f"Here are some notes from my coding journal:\n\n{context}\n\nBased on these, answer this question:\n{question}"}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post(LOCAL_LLM_ENDPOINT, headers=headers, json=payload)
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ Error calling local LLM: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="Coding Journal Assistant", page_icon="📒")
st.title("🔎 Search Your Coding Journal")

query = st.text_input("Ask a question about your coding journal:")

if query:
    # Embed the query
    query_vector = model.encode(["passage: " + query])[0]

    # Search LanceDB
    results = table.search(query_vector).limit(5).to_pandas()

    st.subheader("🔎 Top Journal Entries")
    combined_context = ""
    for i, row in results.iterrows():
        st.markdown(f"**Result {i+1}:**")
        st.write(row["text"])
        st.divider()
        combined_context += row["text"] + "\n\n"

    # Ask local LLM for help interpreting
    st.subheader("🧠 AI Assistant Response")
    answer = call_local_llm(combined_context, query)
    st.markdown(answer)