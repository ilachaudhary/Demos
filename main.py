from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mock_rag import get_rag_response

app = FastAPI(title="Zero Data AI-Finance Platform Demo")

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Versioned endpoints
@app.get("/v1/query", response_class=PlainTextResponse)
def query_v1(text: str):
    response = get_rag_response(text)
    return response


@app.get("/v1.1/query", response_class=PlainTextResponse)
def query_v1_1(text: str):
    # Could simulate improved RAG, multi-modal, etc.
    response = get_rag_response(text) + " (v1.1)"
    return response


@app.get("/v1.2/query", response_class=PlainTextResponse)
def query_v1_2(text: str):
    # Latest improvements
    response = get_rag_response(text) + " (v1.2)"
    return response
