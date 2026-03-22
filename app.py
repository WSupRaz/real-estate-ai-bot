import streamlit as st
import requests
import os
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load env
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")

# OpenRouter
url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Real Estate AI Bot"
}

st.set_page_config(page_title="Real Estate AI Bot", layout="wide")

st.title("🏡 Real Estate AI Assistant")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Load properties
def load_properties():
    with open("properties.txt", "r") as f:
        return f.readlines()

properties = load_properties()

# Create embeddings
@st.cache_data
def create_index(data):
    embeddings = model.encode(data)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = create_index(properties)

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Show chat
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
query = st.chat_input("Ask about properties...")

if query:
    st.session_state.chat.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    # Embed query
    q_embedding = model.encode([query])

    # Search
    D, I = index.search(np.array(q_embedding), 3)
    context = "\n".join([properties[i] for i in I[0]])

    # Prompt
    prompt = f"""
You are a professional real estate agent.

Use this data:
{context}

Answer user query:
{query}

Be helpful, suggest options, sound like a human agent.
"""

    data = {"model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    if "choices" in result:
        answer = result["choices"][0]["message"]["content"]

        st.session_state.chat.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.write(answer)
    else:
        st.error(result)