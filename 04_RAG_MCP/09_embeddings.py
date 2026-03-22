from sentence_transformers import SentenceTransformer
import numpy as np

# ── What is an embedding? ─────────────────────────────────────────────────────
# An embedding is a list of numbers that represents the *meaning* of a piece of text.
# Similar meanings = similar numbers = close together in vector space.
# This is how semantic search works — you search by meaning, not keywords.

# Load a small, fast embedding model (downloads once, ~90MB)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")

# ── Example: embed some sentences ────────────────────────────────────────────
sentences = [
    "How much does the product cost?",        # pricing question
    "What are the pricing plans?",            # also pricing — different words
    "How do I reset my password?",            # account question
    "I forgot my login credentials",          # also account — different words
    "What features are included in the plan?" # features question
]

embeddings = model.encode(sentences)
print(f"Each sentence becomes a vector of {len(embeddings[0])} numbers\n")

# ── Measure similarity between sentences ─────────────────────────────────────
# Cosine similarity: 1.0 = identical meaning, 0.0 = unrelated
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = "What is the price?"
query_embedding = model.encode([query])[0]

print(f"Query: '{query}'\n")
print("Similarity scores:")
for sentence, embedding in zip(sentences, embeddings):
    score = cosine_similarity(query_embedding, embedding)
    bar = "█" * int(score * 30)
    print(f"  {score:.3f} {bar} '{sentence}'")

# Notice: "What are the pricing plans?" scores high even though
# it doesn't contain the word "price" — that's semantic search.
print("\nNotice how meaning matters more than exact word matches.")
print("This is the core idea behind RAG.")
