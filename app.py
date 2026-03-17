import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = []

# Function: Get embedding
def get_embedding(text):
    return model.encode(text)

# Function: Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Page config
st.set_page_config(page_title="AI Memory Engine", page_icon="🧠")

# Sidebar (NEW FEATURE)
st.sidebar.title("📌 About Project")
st.sidebar.write("AI Memory Engine tracks user queries and learns interests using semantic similarity.")
st.sidebar.write(f"📈 Total Queries: {len(st.session_state.memory)}")

# Clear memory button (NEW FEATURE)
if st.sidebar.button("🗑 Clear Memory"):
    st.session_state.memory = []
    st.sidebar.success("Memory cleared!")

# Title
st.title("🧠 AI Memory Engine with Semantic Learning & Interest Detection")
st.write("🚀 This system remembers your queries and understands your interests over time.")

# Input
user_input = st.text_input("💬 Ask something:")

if user_input:
    query_embedding = get_embedding(user_input)

    memory = st.session_state.memory
    results = []

    # Find similar queries
    for item in memory:
        score = cosine_similarity(query_embedding, item["embedding"])
        results.append((item["query"], score))

    results.sort(key=lambda x: x[1], reverse=True)

    # Store current query
    memory.append({
        "query": user_input,
        "embedding": query_embedding
    })

    # Smart AI Response
    if results and results[0][1] > 0.7:
        response = f"🧠 You previously explored **'{results[0][0]}'**.\n\nThis question seems related. I can connect both topics for better understanding."
    else:
        response = "✨ This looks like a new topic. Let's explore it together!"

    # Display response
    st.markdown("### 🤖 AI Response")
    st.write(response)

    # Similar queries
    if results:
        st.markdown("### 🔍 Similar Past Queries")
        for q, score in results[:3]:
            st.write(f"👉 {q}  |  Similarity: {score:.2f}")

    # Interest detection (NEW FEATURE)
    if len(memory) > 2:
        st.markdown("### 📊 Your Learning Interests")
        topics = [item["query"] for item in memory]
        st.write(", ".join(topics[:5]))

    # Full memory view
    with st.expander("📚 View Full Memory"):
        for item in memory:
            st.write(item["query"])