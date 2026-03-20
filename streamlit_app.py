import streamlit as st
from mock_rag import get_rag_response

st.title("Zero Data AI-Finance Demo")
st.info(
    """
    This prototype simulates an AI system that provides financial insights 
    without exposing sensitive data.
    
    The demo uses a simplified Retrieval-Augmented Generation (RAG) workflow.
    """
)

# Suggested questions
st.markdown("### Example questions you can try")
st.markdown("#### For a Loan Applicant:")

st.markdown("""
* "My credit score is 620. Can I get a loan?"
* "What is the minimum credit score required for approval?"
* "What happens if I have existing debt?"
* "How can I improve my chances of getting approved?"
""")

st.markdown("#### For a Loan Officer:")

st.markdown("""
* "Is a customer with a debt-to-income ratio above 40% high risk?"
* "What verification is required for a first-time borrower?"
* "What are the conditions for approving a personal loan?"
* "Does a bankruptcy in the last 3 years disqualify an applicant?"
""")

st.markdown("#### For a Compliance Officer:")

st.markdown("""
* "Can customer financial data leave the system?"
* "Is this platform GDPR compliant?"
* "What happens to query data after a session ends?"
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
