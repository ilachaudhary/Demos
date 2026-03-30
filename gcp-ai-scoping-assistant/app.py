import streamlit as st
from google import genai
import chromadb
import os
from dotenv import load_dotenv
from pathlib import Path

st.components.v1.html("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-CT99WFEGDH"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-CT99WFEGDH');
</script>
""", height=0)

# Load API key
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Gemini embedding function
def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


# Load and chunk docs
@st.cache_resource
def load_docs():
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection("gcp_docs")


    docs_path = Path(__file__).parent / "docs"
    chunks = []
    metadatas = []

    for file in docs_path.glob("*.txt"):
        text = file.read_text()
        words = text.split()
        chunk_size = 100
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            metadatas.append({"source": file.name})

    if chunks:
        embeddings = [get_embedding(chunk) for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
    return collection


# App header
st.title("🚀 GCP AI Scoping Assistant")
st.subheader("For Enterprise Solutions Architects & IT Decision Makers")
st.write("Describe your enterprise AI challenge and get instant Google Cloud recommendations.")

# Example prompts
st.info("""
**Try asking:**
- We are a 500-person retail company needing fraud detection for payments
- We are a bank looking to automate loan document processing
- We need real-time risk scoring for trading transactions
""")

# Industry and company size selectors
col1, col2 = st.columns(2)

with col1:
    industry = st.selectbox(
        "Select your industry:",
        [
            "Select industry...",
            "Retail & E-commerce",
            "Banking & Financial Services",
            "Insurance",
            "Healthcare",
            "Manufacturing",
            "Telecommunications",
            "Energy & Utilities"
        ]
    )

with col2:
    company_size = st.selectbox(
        "Select company size:",
        [
            "Select size...",
            "Startup (1-50 employees)",
            "Mid-size (51-500 employees)",
            "Large Enterprise (500-5000 employees)",
            "Fortune 500 (5000+ employees)"
        ]
    )

# User input
user_input = st.text_area(
    "Describe your enterprise AI use case:",
    placeholder="e.g. We are a large retail company like Nike looking to automate fraud detection...",
    height=150
)

if st.button("Get GCP Recommendations"):
    if user_input:
        with st.spinner("Searching GCP documentation and generating recommendations..."):

            # Initialize defaults
            retrieved_chunks = "No documentation context available."
            source_files = []

            try:
                # Load docs
                collection = load_docs()

                # Embed user question
                query_embedding = get_embedding(user_input)

                # Retrieve relevant chunks
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5
                )

                retrieved_chunks = "\n\n".join(results['documents'][0])
                sources = results['metadatas'][0] if results['metadatas'][0] else []
                source_files = list(set([
                    s.get('source', 'GCP Documentation')
                    for s in sources
                ]))

            except Exception as e:
                st.error(f"RAG error: {e}")

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

First use this context from Google Cloud documentation:
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

Industry: {industry}
Company Size: {company_size}
User question: {user_input}
"""

            # Generate response
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            # Display response
            if "outside the scope" in response.text:
                st.warning(response.text)
            else:
                st.success("✅ Here are your GCP recommendations:")
                st.markdown(response.text)
                # Human Feedback Collection (RLHF Layer)
                st.divider()
                st.subheader("📊 Was this recommendation helpful?")
                st.write("Your feedback helps improve future recommendations.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("👍 Yes, helpful"):
                        import csv
                        import datetime
                        with open("feedback.csv", "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.datetime.now(),
                                industry,
                                company_size,
                                user_input[:100],
                                "positive"
                            ])
                        st.session_state["feedback_given"] = "positive"
                
                with col2:
                    if st.button("👎 Not helpful"):
                        import csv
                        import datetime
                        with open("feedback.csv", "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.datetime.now(),
                                industry,
                                company_size,
                                user_input[:100],
                                "negative"
                            ])
                        st.session_state["feedback_given"] = "negative"
                
                # Show confirmation outside the columns
                if st.session_state.get("feedback_given") == "positive":
                    st.success("Thank you! Feedback recorded.")
                elif st.session_state.get("feedback_given") == "negative":
                    st.warning("Thank you! We'll use this to improve.")

                # Show sources
                st.divider()
                st.subheader("📚 Sources Used")
                if source_files:
                    for source in source_files:
                        st.write(f"• {source}")
                else:
                    st.write("• GCP Official Documentation")
                    st.write("• Gemini AI Knowledge Base")

    else:
        st.warning("Please describe your enterprise AI use case first.")
