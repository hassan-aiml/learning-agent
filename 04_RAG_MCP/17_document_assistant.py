import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="centered"
)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_claude_client():
    return Anthropic()

model  = load_embedding_model()
client = load_claude_client()


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Build KB from my_docs/ folder (Hassan default) ───────────────────────────
@st.cache_resource
def build_default_kb(folder: str = "my_docs") -> tuple:
    """Loads Hassan's resume from my_docs/ at startup. Cached for the session."""
    if not os.path.exists(folder):
        return None, 0, []

    files = [f for f in os.listdir(folder) if f.endswith((".txt", ".md"))]
    if not files:
        return None, 0, []

    all_ids = []
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

    chroma_client = chromadb.PersistentClient(path="./knowledge_base")
    try:
        chroma_client.delete_collection("default_docs")
    except:
        pass
    collection     = chroma_client.create_collection("default_docs")
    all_embeddings = model.encode(all_texts, show_progress_bar=False).tolist()
    collection.add(ids=all_ids, embeddings=all_embeddings, documents=all_texts)

    return collection, len(all_texts), filenames


# ── Build KB from uploaded file ───────────────────────────────────────────────
def build_uploaded_kb(uploaded_file) -> tuple:
    """Builds a fresh KB from a user-uploaded file. Not cached — rebuilds each time."""
    text = uploaded_file.read().decode("utf-8").strip()
    if not text:
        return None, 0

    chunks = chunk_text(text)
    all_ids        = [f"upload_chunk_{i}" for i in range(len(chunks))]
    all_embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    chroma_client = chromadb.PersistentClient(path="./knowledge_base")
    try:
        chroma_client.delete_collection("uploaded_docs")
    except:
        pass
    collection = chroma_client.create_collection("uploaded_docs")
    collection.add(ids=all_ids, embeddings=all_embeddings, documents=chunks)

    return collection, len(chunks)


# ── Derive a readable title from a filename ───────────────────────────────────
def title_from_filename(filename: str) -> str:
    """
    Turns a filename into a friendly header title.
    e.g. 'company_faq.txt' → 'Chat with: company faq'
         'ProductManual_v2.md' → 'Chat with: ProductManual v2'
    """
    name = os.path.splitext(filename)[0]          # strip extension
    name = name.replace("_", " ").replace("-", " ")
    return f"💬 Chat with: {name}"


# ── RAG retrieval ─────────────────────────────────────────────────────────────
def retrieve_context(collection, query: str, top_k: int = 4) -> str:
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    chunks  = results["documents"][0]
    return "\n\n".join([f"[Document {i+1}]: {chunk}" for i, chunk in enumerate(chunks)])


# ── System prompts ────────────────────────────────────────────────────────────
HASSAN_SYSTEM_PROMPT = """You are a professional HR assistant helping recruiters and 
hiring managers learn about Hassan M. Hai from his resume.

IMPORTANT RULES:
- Only answer using the resume context provided — never invent experience or skills
- If the resume doesn't contain enough information to answer, say so clearly
- Refer to the candidate by first name: Hassan
- Be objective, professional, and concise
- When quoting specific facts (years, companies, technologies), be precise
- Do not speculate about things not mentioned in the resume"""

def uploaded_system_prompt(filename: str) -> str:
    name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
    return f"""You are a helpful assistant answering questions about the document: '{name}'.

IMPORTANT RULES:
- Only answer using the document context provided — never invent information
- If the document doesn't contain enough information to answer, say so clearly
- Be concise, accurate, and helpful
- When quoting specific facts, be precise
- Do not speculate about things not mentioned in the document"""


# ── Load default KB (Hassan) at startup ──────────────────────────────────────
default_collection, default_chunks, default_files = build_default_kb("my_docs")


# ── Session state ─────────────────────────────────────────────────────────────
if "messages"           not in st.session_state:
    st.session_state.messages           = []
if "uploaded_collection" not in st.session_state:
    st.session_state.uploaded_collection = None
if "uploaded_filename"   not in st.session_state:
    st.session_state.uploaded_filename   = None
if "uploaded_chunks"     not in st.session_state:
    st.session_state.uploaded_chunks     = 0


# ── Determine active mode ─────────────────────────────────────────────────────
using_upload = st.session_state.uploaded_collection is not None
active_collection = st.session_state.uploaded_collection if using_upload else default_collection
active_system_prompt = (
    uploaded_system_prompt(st.session_state.uploaded_filename)
    if using_upload
    else HASSAN_SYSTEM_PROMPT
)


# ── Dynamic header ────────────────────────────────────────────────────────────
if using_upload:
    st.title(title_from_filename(st.session_state.uploaded_filename))
    st.caption(f"Answering questions from: **{st.session_state.uploaded_filename}**")
else:
    st.title("🧑‍💼 Learn about Hassan")
    st.caption("Ask anything about Hassan's experience, skills, or background.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Default KB status ─────────────────────────────────────────────────────
    st.header("Default Document")
    if default_collection:
        st.success("Hassan's resume loaded")
        st.metric("Chunks indexed", default_chunks)
    else:
        st.error("No default resume found in `my_docs/`")

    st.divider()

    # ── Upload your own document ──────────────────────────────────────────────
    st.header("Upload Your Own Document")
    st.info(
        "Upload a plain text file to chat with your own document instead.\n\n"
        "**Accepted formats:** `.txt` or `.md`\n\n"
        "**How to prepare your file:**\n"
        "- Save your document as a `.txt` or `.md` file\n"
        "- Make sure it is plain text (no PDF, Word, or Excel)\n"
        "- Any document works: resume, FAQ, manual, notes"
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "md"],
        help="Only .txt and .md files are supported."
    )

    if uploaded_file:
        if st.button("Load document", type="primary", use_container_width=True):
            with st.spinner("Building knowledge base from your document..."):
                collection, total = build_uploaded_kb(uploaded_file)
            if collection:
                st.session_state.uploaded_collection = collection
                st.session_state.uploaded_filename   = uploaded_file.name
                st.session_state.uploaded_chunks     = total
                st.session_state.messages            = []  # reset chat
                st.success(f"Loaded! {total} chunks indexed.")
                st.rerun()
            else:
                st.error("Could not read the file. Make sure it contains plain text.")

    # Show uploaded doc status
    if using_upload:
        st.divider()
        st.caption(f"Active upload: **{st.session_state.uploaded_filename}**")
        st.metric("Upload chunks", st.session_state.uploaded_chunks)
        if st.button("↩️ Switch back to Hassan's resume", use_container_width=True):
            st.session_state.uploaded_collection = None
            st.session_state.uploaded_filename   = None
            st.session_state.uploaded_chunks     = 0
            st.session_state.messages            = []
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    show_context = st.toggle("Show retrieved context", value=False)
    st.caption("See which document sections were used for each answer.")


# ── Guard: no active collection ───────────────────────────────────────────────
if not active_collection:
    st.error("No document loaded.")
    st.info("Add Hassan's resume to `my_docs/` or upload a document in the sidebar.")
    st.stop()


# ── Chat input placeholder ────────────────────────────────────────────────────
chat_placeholder = (
    f"Ask about {os.path.splitext(st.session_state.uploaded_filename)[0]}..."
    if using_upload
    else "Ask about Hassan..."
)


# ── Render existing messages ──────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input(chat_placeholder):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # RAG retrieval
    context = retrieve_context(active_collection, prompt)

    if show_context:
        with st.expander("📄 Retrieved document sections", expanded=False):
            st.text(context)

    # Augment with context
    doc_label = st.session_state.uploaded_filename if using_upload else "Hassan's resume"
    augmented_message = f"""Here is relevant information from {doc_label}:

{context}

---
Question: {prompt}"""

    # Clean conversation history
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
            system=active_system_prompt,
            messages=history
        ) as stream:
            for chunk in stream.text_stream:
                full_response += chunk
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    chunk_count = len(context.split("[Document")) - 1
    st.caption(f"Retrieved {chunk_count} document sections")
