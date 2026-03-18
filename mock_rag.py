from sentence_transformers import SentenceTransformer
import numpy as np

# Load small embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Mock corpus for financial AI responses
mock_corpus = [
    "Loan approved if credit > 670",
    "High risk loan flagged",
    "Customer needs additional verification"
]

# Precompute embeddings
embeddings = model.encode(mock_corpus)


def get_rag_response(query: str) -> str:
    """
    Simple vector similarity to choose the best mock response.
    """
    query_emb = model.encode([query])
    similarity = np.dot(embeddings, query_emb[0]) / (np.linalg.norm(embeddings, axis=1))
    best_idx = int(np.argmax(similarity))
    return mock_corpus[best_idx]
