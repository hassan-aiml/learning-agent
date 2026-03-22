import chromadb
from sentence_transformers import SentenceTransformer

# ── Build a local vector knowledge base from documents ────────────────────────
# In a real product, these would be loaded from .txt, .pdf, or .md files.
# Here we use hardcoded text so you can run it immediately.

# Pretend these are chunks from a company's documentation
DOCUMENTS = [
    # Pricing
    {"id": "doc1", "text": "Our Starter plan costs $29/month and includes up to 3 users, 10GB storage, and email support."},
    {"id": "doc2", "text": "The Pro plan is $79/month and includes unlimited users, 100GB storage, priority support, and API access."},
    {"id": "doc3", "text": "Enterprise pricing is custom — contact sales@company.com for a quote based on your team size."},

    # Features
    {"id": "doc4", "text": "The dashboard lets you track all your projects in one place with Kanban boards and Gantt charts."},
    {"id": "doc5", "text": "You can integrate with Slack, Notion, Google Drive, and over 50 other tools via our integrations page."},
    {"id": "doc6", "text": "The API supports REST and GraphQL. Full documentation is available at docs.company.com/api."},

    # Support
    {"id": "doc7", "text": "To reset your password, go to the login page and click 'Forgot password'. A reset link will be sent to your email."},
    {"id": "doc8", "text": "Support hours are Monday to Friday, 9am to 6pm EST. Pro and Enterprise users get 24/7 support."},
    {"id": "doc9", "text": "You can reach our support team via live chat in the app, or by emailing support@company.com."},

    # Onboarding
    {"id": "doc10", "text": "Getting started takes about 10 minutes. After signing up, follow the onboarding checklist in your dashboard."},
    {"id": "doc11", "text": "We offer free onboarding calls for Pro and Enterprise customers. Book one at company.com/onboarding."},
]

# ── Step 1: Load embedding model ──────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Step 2: Create a local ChromaDB database ──────────────────────────────────
# persist_directory means it saves to disk — survives between runs
chroma_client = chromadb.PersistentClient(path="./knowledge_base")

# Delete collection if it exists (so we can re-run cleanly)
try:
    chroma_client.delete_collection("company_docs")
except:
    pass

collection = chroma_client.create_collection(
    name="company_docs",
    metadata={"description": "Company product documentation"}
)

# ── Step 3: Embed all documents and store in ChromaDB ─────────────────────────
print("Embedding and storing documents...")
texts = [doc["text"] for doc in DOCUMENTS]
ids   = [doc["id"]   for doc in DOCUMENTS]

embeddings = model.encode(texts).tolist()  # ChromaDB needs a list, not numpy array

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=texts
)

print(f"Stored {collection.count()} documents in the knowledge base.\n")

# ── Step 4: Test retrieval ────────────────────────────────────────────────────
def search(query: str, top_k: int = 3):
    """Search the knowledge base for the most relevant chunks."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]  # list of matching text chunks

# Try some queries
test_queries = [
    "How much does it cost?",
    "I can't log in to my account",
    "Does it work with Slack?",
    "How do I get help on weekends?"
]

for query in test_queries:
    print(f"Query: '{query}'")
    chunks = search(query)
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. {chunk}")
    print()

print("Knowledge base built and saved to ./knowledge_base/")
print("Run 10_rag_chatbot.py next to chat with it.")
