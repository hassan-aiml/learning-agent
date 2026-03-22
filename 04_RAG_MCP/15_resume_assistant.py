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

st.title("🧑‍💼 Resume Assistant")
st.caption("Upload a resume and ask anything about the candidate's experience, skills, or background.")


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
    """
    Split text into overlapping chunks.
    Smaller chunk_size (300) works better for resumes than long documents
    because resume sections are short and dense with specific facts.
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Build knowledge base (from script 12) ────────────────────────────────────
def build_knowledge_base(uploaded_files) -> tuple:
    chroma_client = chromadb.PersistentClient(path="./knowledge_base")
    try:
        chroma_client.delete_collection("resume_docs")
    except:
        pass

    collection = chroma_client.create_collection("resume_docs")

    all_ids   = []
    all_texts = []

    progress = st.progress(0, text="Processing resume...")

    for file_index, uploaded_file in enumerate(uploaded_files):
        text   = uploaded_file.read().decode("utf-8").strip()
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_ids.append(f"{uploaded_file.name}_chunk_{i}")
            all_texts.append(chunk)

        progress.progress(
            (file_index + 1) / len(uploaded_files),
            text=f"Chunked: {uploaded_file.name} → {len(chunks)} chunks"
        )

    progress.progress(0.8, text="Embedding chunks...")
    all_embeddings = model.encode(all_texts, show_progress_bar=False).tolist()

    progress.progress(0.95, text="Storing in knowledge base...")
    collection.add(ids=all_ids, embeddings=all_embeddings, documents=all_texts)

    progress.progress(1.0, text="Done!")
    return collection, len(all_texts)


# ── Load existing KB ──────────────────────────────────────────────────────────
def load_existing_kb():
    try:
        chroma_client = chromadb.PersistentClient(path="./knowledge_base")
        collection    = chroma_client.get_collection("resume_docs")
        return collection
    except:
        return None


# ── RAG retrieval (from script 11) ───────────────────────────────────────────
def retrieve_context(collection, query: str, top_k: int = 4) -> str:
    """
    top_k=4 instead of 3 because resumes have many short sections
    and we want to capture more of the relevant detail per question.
    """
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    chunks  = results["documents"][0]
    return "\n\n".join([f"[Document {i+1}]: {chunk}" for i, chunk in enumerate(chunks)])


# ── System prompt — resume-focused ───────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional HR assistant helping recruiters and hiring 
managers learn about a candidate from their resume.

IMPORTANT RULES:
- Only answer using the resume context provided — never invent experience or skills
- If the resume doesn't contain enough information to answer, say so clearly
- Be objective, professional, and concise
- When quoting specific facts (years, companies, technologies), be precise
- If asked for an opinion or recommendation, base it strictly on what the resume shows
- Do not speculate about things not mentioned in the resume"""


# ── Suggested questions ───────────────────────────────────────────────────────
SUGGESTED_QUESTIONS = [
    "What is this candidate's most recent role?",
    "What programming languages do they know?",
    "How many years of experience do they have?",
    "What are their strongest technical skills?",
    "Have they worked with AI or machine learning?",
    "What is their educational background?",
    "Do they have any leadership experience?",
]


# ── Session state ─────────────────────────────────────────────────────────────
if "messages"   not in st.session_state:
    st.session_state.messages   = []
if "collection" not in st.session_state:
    st.session_state.collection = load_existing_kb()
if "resume_name" not in st.session_state:
    st.session_state.resume_name = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Resume")

    uploaded_files = st.file_uploader(
        "Upload a .txt or .md resume file",
        type=["txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Load resume", type="primary", use_container_width=True):
            collection, total = build_knowledge_base(uploaded_files)
            st.session_state.collection  = collection
            st.session_state.messages    = []
            st.session_state.resume_name = ", ".join([f.name for f in uploaded_files])
            st.success(f"Loaded! {total} chunks indexed.")
            st.rerun()

    st.divider()

    if st.session_state.collection:
        st.metric("Chunks indexed", st.session_state.collection.count())
        if st.session_state.resume_name:
            st.caption(f"Active resume: **{st.session_state.resume_name}**")
    else:
        st.warning("No resume loaded yet.")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    show_context = st.toggle("Show retrieved context", value=False)
    st.caption("See which resume sections were used for each answer.")


# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.collection:
    st.info("👈 Upload a resume in the sidebar to get started.")

    with st.expander("How to use this app"):
        st.markdown("""
1. **Upload a resume** as a `.txt` or `.md` file using the sidebar
2. Click **Load resume** to index it
3. **Ask questions** about the candidate in the chat below

The assistant only answers from the resume — it will never make up experience or skills.

**Try asking:**
- What technologies does this candidate know?
- How long have they been working in software?
- Do they have any management experience?
        """)
else:
    # Show suggested questions if conversation is empty
    if not st.session_state.messages:
        st.markdown("**Suggested questions to get started:**")
        cols = st.columns(2)
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 2]:
                if st.button(question, use_container_width=True, key=f"sq_{i}"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()

        st.divider()

    # Render existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the candidate..."):

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # RAG retrieval (script 11 pattern)
        context = retrieve_context(st.session_state.collection, prompt)

        if show_context:
            with st.expander("📄 Retrieved resume sections", expanded=False):
                st.text(context)

        # Augment with context (script 11 pattern)
        augmented_message = f"""Here is relevant information from the candidate's resume:

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
