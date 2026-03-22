import os
import chromadb
from sentence_transformers import SentenceTransformer

# ── Load YOUR OWN documents into the knowledge base ──────────────────────────
# This script reads .txt and .md files from a folder called 'my_docs/'
# and loads them into the vector database.
#
# HOW TO USE:
# 1. Create a folder called 'my_docs' in your project directory
# 2. Drop any .txt or .md files in there (company FAQs, product docs, etc.)
# 3. Run this script
# 4. Then run 11_rag_chatbot.py to chat with your documents

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Why overlap? So that information spanning two chunks isn't lost.
    e.g. a sentence that starts at the end of chunk 1 and ends at the start
    of chunk 2 will still be captured by the overlap.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward but keep some overlap

    return chunks


def load_documents(folder: str) -> list[dict]:
    """Load all .txt and .md files from a folder and split into chunks."""
    documents = []
    supported = (".txt", ".md")

    if not os.path.exists(folder):
        print(f"Folder '{folder}' not found. Creating it...")
        os.makedirs(folder)
        print(f"Add your .txt or .md files to '{folder}/' and re-run.\n")
        return []

    files = [f for f in os.listdir(folder) if f.endswith(supported)]

    if not files:
        print(f"No .txt or .md files found in '{folder}/'")
        print("Add your documents and re-run.\n")
        return []

    for filename in files:
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{filename}_chunk_{i}",
                "text": chunk,
                "source": filename
            })
        print(f"  Loaded '{filename}' → {len(chunks)} chunks")

    return documents


# ── Main ingestion pipeline ───────────────────────────────────────────────────
print("Document ingestion pipeline\n")

docs = load_documents("my_docs")

if docs:
    print(f"\nTotal chunks to embed: {len(docs)}")
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Rebuild the knowledge base with your documents
    chroma_client = chromadb.PersistentClient(path="./knowledge_base")

    try:
        chroma_client.delete_collection("company_docs")
    except:
        pass

    collection = chroma_client.create_collection("company_docs")

    print("Embedding and storing chunks...")
    texts = [doc["text"] for doc in docs]
    ids   = [doc["id"]   for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(ids=ids, embeddings=embeddings, documents=texts)

    print(f"\nDone. {collection.count()} chunks stored in knowledge base.")
    print("Run 11_rag_chatbot.py to chat with your documents.")
