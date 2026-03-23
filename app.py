import streamlit as st
import requests
import os
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Real Estate AI Bot", layout="wide")

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")

if not API_KEY:
    st.error("API key not found. Please check .env or Streamlit secrets.")
    st.stop()

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Real Estate AI Bot"
}

# ---------------- UI ----------------
st.title("🏡 Real Estate AI Assistant")
st.markdown("Find your dream property instantly with AI 🏠")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("Loading AI model..."):
    model = load_model()

# ---------------- LOAD DATA ----------------
def load_properties():
    try:
        with open("properties.txt", "r") as f:
            return f.readlines()
    except:
        return [
            "2BHK in Pune - 45 Lakhs - Wakad - Near IT Park",
            "3BHK in Pune - 75 Lakhs - Hinjewadi - IT hub",
            "1BHK in Pune - 25 Lakhs - Pimpri - Budget option",
            "2BHK in Mumbai - 90 Lakhs - Thane - Good connectivity",
            "3BHK in Bangalore - 80 Lakhs - Whitefield - IT hub"
        ]

properties = load_properties()

# ---------------- CREATE INDEX ----------------
@st.cache_data
def create_index(data):
    embeddings = model.encode(data)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

index = create_index(properties)

# ---------------- CHAT ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask about properties...")

if query:
    st.session_state.chat.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    # Embed query
    q_embedding = model.encode([query])

    # Search top results
    _, I = index.search(np.array(q_embedding), 3)

    context = "\n".join([properties[i] for i in I[0] if i < len(properties)])

    # Prompt
    prompt = f"""
You are a professional real estate agent.

Available properties:
{context}

User query:
{query}

Answer naturally, suggest best options, and encourage a visit or call.
"""

    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    # API call (safe)
    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    # Response handling
    if "choices" in result:
        answer = result["choices"][0]["message"]["content"]

        st.session_state.chat.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.write(answer)

            st.markdown("👉 Interested? Request a callback for site visit!")
    else:
        st.error(result)