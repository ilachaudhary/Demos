import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from pathlib import Path

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load and chunk docs
@st.cache_resource
def load_docs():
    client = chromadb.Client()
    collection = client.create_collection("gcp_docs")
    embedder = load_embedder()

    docs_path = Path("docs")
    chunks = []

    for file in docs_path.glob("*.txt"):
        text = file.read_text()
        # Split into chunks of ~500 characters
        words = text.split()
        chunk_size = 100  # words per chunk
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)

    if chunks:
        embeddings = embedder.encode(chunks).tolist()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )

    return collection, embedder

# App header
st.title("👨‍💼 GCP AI Scoping Assistant")
st.subheader("For Enterprise Solutions Architects & IT Decision Makers")
st.write("Describe your enterprise AI challenge and get instant Google Cloud recommendations.")

# Example prompts
st.info("""
**Try asking:**
- We are a 500-person retail company needing fraud detection for payments
- We are a bank looking to automate loan document processing
- We need real-time risk scoring for trading transactions
""")

# User input
user_input = st.text_area(
    "Describe your enterprise AI use case:",
    placeholder="e.g. We are a large retail company like Nike looking to automate fraud detection...",
    height=150
)

if st.button("Get GCP Recommendations"):
    if user_input:
        with st.spinner("Searching GCP documentation and generating recommendations..."):

            try:
                # Load docs and embedder
                collection, embedder = load_docs()

                # Embed user question
                query_embedding = embedder.encode([user_input]).tolist()[0]

                # Retrieve relevant chunks
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5
                )

                retrieved_chunks = "\n\n".join(results['documents'][0])

            except Exception as e:
                retrieved_chunks = "No documentation context available."

            # Build prompt
            prompt = f"""
You are an expert Google Cloud Solutions Architect 
with deep knowledge of enterprise AI architecture.

STRICT RULE: You only answer questions about:
- Google Cloud AI services and architecture
- Enterprise AI project scoping and recommendations  
- AI/ML implementation strategies for businesses

If the question is outside these topics, respond with:
"⚠️ This question is outside the scope of the GCP AI Scoping Assistant. 
This tool is designed exclusively to help enterprise Solutions Architects 
and IT Decision Makers scope AI projects on Google Cloud. 
Please describe your enterprise AI use case and I'll provide 
tailored GCP service recommendations."

First, use this context from Google Cloud documentation:
{retrieved_chunks}

Then use your own knowledge to:
- Fill gaps the docs don't cover
- Reason about tradeoffs between approaches
- Provide additional relevant recommendations
- Compare alternatives where helpful

Always prioritize the provided GCP documentation 
but reason beyond it when it adds value.

Provide a structured recommendation including:
1. Top 3 recommended Google Cloud AI services
2. Suggested architecture in plain language
3. Complexity rating (Simple / Medium / Complex)
4. Compliance considerations (GDPR, SOC 2, PCI-DSS)
5. First step they should take this week

User question: {user_input}
"""
            # Generate response
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            # Check if response is out of scope

            st.success("✅ Here are your GCP recommendations:")
            st.markdown(response.text)
    else:
        st.warning("Please describe your enterprise AI use case first.")
