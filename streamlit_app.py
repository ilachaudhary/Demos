import streamlit as st
from mock_rag import get_rag_response

st.title("Zero Data AI-Finance Demo 1")
st.info(
    """
    This prototype simulates an AI system that provides financial insights 
    without exposing sensitive data.
    
    The demo uses a simplified Retrieval-Augmented Generation (RAG) workflow.
    """
)

# Suggested questions
st.markdown("### Example questions you can try")

st.markdown("""
• Is this loan high risk?  
• Does this customer need verification?  
• What happens if credit score is low?  
• Can financial data leave the system?  
""")
# Store conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Input form (resets after submit)
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_input("Enter query:")
    submit_button = st.form_submit_button("Ask")

if submit_button and user_input:
    # Call FastAPI endpoint
    response = get_rag_response(user_input)
    st.session_state.history.append((user_input, response))

# Show chat history
for i, (query, resp) in enumerate(st.session_state.history):
    st.markdown(f"**Query {i+1}:** {query}")
    st.markdown(f"**Response:** {resp}")
    st.markdown("---")
