import streamlit as st
import requests

st.title("Zero Data AI-Finance Demo")

# Store conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Input form (resets after submit)
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_input("Enter query:")
    submit_button = st.form_submit_button("Ask")

if submit_button and user_input:
    # Call FastAPI endpoint
    res = requests.get(f"http://127.0.0.1:8000/v1.2/query?text={user_input}")
    response_text = res.text  # plain text
    st.session_state.history.append((user_input, response_text))

# Show chat history
for i, (query, resp) in enumerate(st.session_state.history):
    st.markdown(f"**Query {i+1}:** {query}")
    st.markdown(f"**Response:** {resp}")
    st.markdown("---")