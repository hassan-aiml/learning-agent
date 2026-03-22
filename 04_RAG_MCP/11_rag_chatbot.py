import os
import chromadb
from dotenv import load_dotenv
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Anthropic()

# ── Load the knowledge base built in the previous script ─────────────────────
print("Loading knowledge base and embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./knowledge_base")
collection = chroma_client.get_collection("company_docs")
print(f"Loaded {collection.count()} documents.\n")


# ── System prompt — strict grounding in documents ────────────────────────────
SYSTEM_PROMPT = """You are a helpful customer support assistant for our software product.

IMPORTANT RULES:
- Only answer questions using the context provided to you
- If the context doesn't contain enough information to answer, say so honestly
- Never make up prices, features, or policies
- Be friendly, concise, and direct
- If the user asks something unrelated to the product, politely redirect them"""


# ── RAG retrieval function ────────────────────────────────────────────────────
def retrieve_context(query: str, top_k: int = 3) -> str:
    """Find the most relevant document chunks for a query."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    chunks = results["documents"][0]
    # Format chunks clearly for Claude
    return "\n\n".join([f"[Document {i+1}]: {chunk}" for i, chunk in enumerate(chunks)])


# ── RAG chatbot ───────────────────────────────────────────────────────────────
def chat():
    """
    The RAG pattern in action:
    1. User asks a question
    2. We retrieve relevant chunks from the knowledge base
    3. We inject those chunks into the prompt as context
    4. Claude answers based ONLY on that context
    
    This grounds Claude's responses in your actual documents.
    No hallucination. No made-up prices or features.
    """
    conversation_history = []

    print("Customer Support Chatbot")
    print("Powered by Claude + RAG")
    print("Type 'quit' to exit\n")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        # ── The RAG step: retrieve relevant context ───────────────────────────
        context = retrieve_context(user_input)

        # ── Inject context into the user message ──────────────────────────────
        # We add the retrieved docs as context before the actual question.
        # Claude uses this context to form a grounded answer.
        augmented_message = f"""Here is relevant information from our documentation:

{context}

---
Customer question: {user_input}"""

        # Add to conversation history
        conversation_history.append({
            "role": "user",
            "content": augmented_message
        })

        # ── Send to Claude ────────────────────────────────────────────────────
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=conversation_history
        )

        assistant_reply = response.content[0].text

        # Store Claude's reply in history (without the context injection)
        # We swap the augmented message for the clean original so history stays readable
        conversation_history[-1]["content"] = user_input
        conversation_history.append({
            "role": "assistant",
            "content": assistant_reply
        })

        print(f"\nAssistant: {assistant_reply}")

        # Show which docs were retrieved (useful for debugging)
        print(f"\n[Retrieved {len(context.split('[Document'))-1} document chunks]")


if __name__ == "__main__":
    chat()
