import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Assistant",
    page_icon="🧑‍💼",
    layout="centered"
)

st.title("🧑‍💼 Learn about Hassan")
st.caption("Ask anything about Hassan's experience, skills, or background.")


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_claude_client():
    return Anthropic()

model  = load_embedding_model()
client = load_claude_client()


# ── Chunking (from script 12) ─────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Load docs from my_docs/ and build KB (from script 12) ────────────────────
@st.cache_resource
def build_knowledge_base_from_folder(folder: str = "my_docs") -> tuple:
    """
    Reads all .txt and .md files from the my_docs/ folder,
    chunks and embeds them, stores in ChromaDB.
    Cached so it only runs once per session.
    """
    if not os.path.exists(folder):
        return None, 0, []

    files = [f for f in os.listdir(folder) if f.endswith((".txt", ".md"))]
    if not files:
        return None, 0, []

    all_ids   = []
    all_texts = []
    filenames = []

    for filename in files:
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_ids.append(f"{filename}_chunk_{i}")
            all_texts.append(chunk)
        filenames.append(filename)

    if not all_texts:
        return None, 0, []

    # Build ChromaDB collection
    chroma_client = chromadb.PersistentClient(path="./knowledge_base")
    try:
        chroma_client.delete_collection("resume_docs")
    except:
        pass

    collection     = chroma_client.create_collection("resume_docs")
    all_embeddings = model.encode(all_texts, show_progress_bar=False).tolist()
    collection.add(ids=all_ids, embeddings=all_embeddings, documents=all_texts)

    return collection, len(all_texts), filenames


# ── RAG retrieval (from script 11) ───────────────────────────────────────────
def retrieve_context(collection, query: str, top_k: int = 4) -> str:
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    chunks  = results["documents"][0]
    return "\n\n".join([f"[Document {i+1}]: {chunk}" for i, chunk in enumerate(chunks)])


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional HR assistant helping recruiters and hiring 
managers learn about Hassan M. Hai from his resume.

IMPORTANT RULES:
- Only answer using the resume context provided — never invent experience or skills
- If the resume doesn't contain enough information to answer, say so clearly
- Refer to the candidate by first name: Hassan
- Be objective, professional, and concise
- When quoting specific facts (years, companies, technologies), be precise
- If asked for an opinion or recommendation, base it strictly on what the resume shows
- Do not speculate about things not mentioned in the resume"""


# ── Load knowledge base on startup ───────────────────────────────────────────
collection, total_chunks, loaded_files = build_knowledge_base_from_folder("my_docs")


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Knowledge Base")

    if collection:
        st.success("Resume loaded")
        st.metric("Chunks indexed", total_chunks)
        st.caption("Files loaded from `my_docs/`:")
        for filename in loaded_files:
            st.caption(f"• {filename}")
    else:
        st.error("No resume found in `my_docs/`")
        st.caption("Add a .txt or .md resume file to the `my_docs/` folder and restart the app.")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    show_context = st.toggle("Show retrieved context", value=False)
    st.caption("See which resume sections were used for each answer.")


# ── Main area ─────────────────────────────────────────────────────────────────
if not collection:
    st.error("No resume loaded.")
    st.info("Add your resume as a `.txt` or `.md` file to the `my_docs/` folder, then restart the app.")
    st.stop()

# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about Hassan..."):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # RAG retrieval (script 11 pattern)
    context = retrieve_context(collection, prompt)

    if show_context:
        with st.expander("📄 Retrieved sections from Hassan's resume", expanded=False):
            st.text(context)

    # Augment with context (script 11 pattern)
    augmented_message = f"""Here is relevant information from Hassan's resume:

{context}

---
Question: {prompt}"""

    # Clean conversation history (script 11 pattern)
    history = []
    for msg in st.session_state.messages[:-1]:
        history.append({"role": msg["role"], "content": msg["content"]})
    history.append({"role": "user", "content": augmented_message})

    # Stream response
    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""

        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=history
        ) as stream:
            for chunk in stream.text_stream:
                full_response += chunk
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    # Store clean reply (script 11 pattern)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    chunk_count = len(context.split("[Document")) - 1
    st.caption(f"Retrieved {chunk_count} resume sections")
