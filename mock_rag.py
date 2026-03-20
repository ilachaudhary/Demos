from sentence_transformers import SentenceTransformer
import numpy as np

# Load small embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Mock corpus for financial AI responses
mock_corpus = [
    # Loan Applicant
    "Loan may be approved with higher interest if credit score is 620",
    "Minimum credit score typically 620 for loan eligibility",
    "Existing debt evaluated using debt-to-income ratio",
    "Improve approval chances by increasing credit score and reducing debt",

    # Loan Officer
    "DTI above 40% flagged as high risk loan",
    "First-time borrower requires income and identity verification",
    "Personal loan approval based on credit score income and DTI",
    "Recent bankruptcy triggers additional risk review",

    # Compliance Officer
    "Customer financial data restricted within secure systems",
    "Platform must comply with GDPR data protection rules",
    "User queries stored temporarily for audit and monitoring"

    "Additional verification needed"
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
