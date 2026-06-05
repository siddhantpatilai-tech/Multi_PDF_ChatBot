import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import asyncio
from datetime import datetime

# ── LangChain imports (all corrected for v0.2+) ─────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate          # ✅ fixed
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Fix 3: safe event-loop bootstrap ────────────────────────────────────────
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# ════════════════════════════════════════════════════════════════════════════
# 1. PDF TEXT EXTRACTION
# ════════════════════════════════════════════════════════════════════════════
def get_pdf_text(pdf_docs):
    """Return concatenated text from all uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:          # Fix 4: guard – extract_text() can return None
                text += extracted
    return text


# ════════════════════════════════════════════════════════════════════════════
# 2. TEXT CHUNKING
# ════════════════════════════════════════════════════════════════════════════
def get_text_chunks(text):
    """Split text into overlapping chunks suitable for embedding."""
    # Fix 5: removed model_name param – chunking is model-agnostic here
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # Fix 6: original was 100 – far too small
        chunk_overlap=200,
    )
    return splitter.split_text(text)


# ════════════════════════════════════════════════════════════════════════════
# 3. VECTOR STORE
# ════════════════════════════════════════════════════════════════════════════
def get_embeddings():
    """Return a shared HuggingFace embeddings object."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(text_chunks):
    """Embed chunks and persist to disk as a FAISS index."""
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def load_vector_store():
    """Load the previously saved FAISS index from disk."""
    return FAISS.load_local(
        "faiss_index",
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# 4. QA CHAIN  –  MODERNISED (LCEL, replaces deprecated load_qa_chain)
# ════════════════════════════════════════════════════════════════════════════
def build_qa_chain(api_key: str):
    """
    Build a retrieval-augmented QA chain using LangChain Expression Language.

    Pipeline:
        user question
            → retrieve top-k docs from FAISS
            → format context + question into prompt
            → Gemini LLM
            → parse plain-text answer
    """
    llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3,
    google_api_key=api_key,
)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Answer the question as thoroughly as possible
using ONLY the information in the provided context.
If the answer is not contained in the context, say exactly:
"answer is not available in the context"

Context:
{context}

Question:
{question}

Answer:""",
    )

    # Fix 7: LCEL chain replaces the deprecated load_qa_chain / chain() call pattern
    chain = prompt | llm | StrOutputParser()
    return chain


# ════════════════════════════════════════════════════════════════════════════
# 5. CHAT MESSAGE HTML HELPER
# ════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem;
    margin-bottom: 1rem; display: flex; align-items: flex-start;
}
.chat-message.user { background-color: #2b313e; }
.chat-message.bot  { background-color: #475063; }
.chat-message .avatar { width: 15%; }
.chat-message .avatar img {
    max-width: 60px; max-height: 60px;
    border-radius: 50%; object-fit: cover;
}
.chat-message .message { width: 85%; padding: 0 1rem; color: #fff; }
</style>
"""

USER_AVATAR = "https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"
BOT_AVATAR  = "https://i.ibb.co/wNmYHsx/langchain-logo.webp"


def render_message(question: str, answer: str):
    """Render a single Q&A pair as styled chat bubbles."""
    st.markdown(
        f"""
        <div class="chat-message user">
            <div class="avatar"><img src="{USER_AVATAR}"></div>
            <div class="message">{question}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar"><img src="{BOT_AVATAR}"></div>
            <div class="message">{answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. HANDLE A USER QUESTION
# ════════════════════════════════════════════════════════════════════════════
def handle_question(user_question: str, api_key: str, pdf_docs, conversation_history: list):
    """Process a question against the uploaded PDFs and display the answer."""
    if not api_key:
        st.warning("Please enter your Google API Key.")
        return
    if not pdf_docs:
        st.warning("Please upload at least one PDF file.")
        return

    with st.spinner("Thinking…"):
        # Build fresh index from current PDFs
        raw_text = get_pdf_text(pdf_docs)
        if not raw_text.strip():
            st.error("Could not extract any text from the uploaded PDFs.")
            return

        chunks = get_text_chunks(raw_text)
        build_vector_store(chunks)

        # Retrieve relevant chunks
        db   = load_vector_store()
        docs = db.similarity_search(user_question, k=4)

        # Fix 8: LCEL invocation – pass context as joined doc page_content strings
        context = "\n\n".join(doc.page_content for doc in docs)
        chain   = build_qa_chain(api_key)

        # Fix 9: chain.invoke() replaces the deprecated chain({...}) call
        answer = chain.invoke({"context": context, "question": user_question})

    # Save to history
    pdf_names = ", ".join(pdf.name for pdf in pdf_docs)
    conversation_history.append({
        "question":  user_question,
        "answer":    answer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pdf_name":  pdf_names,
    })

    # ── Inject CSS once, then render all messages newest-first ──────────────
    st.markdown(_CSS, unsafe_allow_html=True)

    # Latest pair first
    render_message(user_question, answer)

    # Older pairs below
    for entry in reversed(conversation_history[:-1]):
        render_message(entry["question"], entry["answer"])

    # ── CSV download in sidebar ──────────────────────────────────────────────
    df  = pd.DataFrame(conversation_history)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.sidebar.markdown(
        f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv">'
        "<button>⬇ Download conversation as CSV</button></a>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# 7. MAIN APP
# ════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.header("Chat with multiple PDFs 📚")

    # Session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # ── Social badges ────────────────────────────────────────────────────────
    st.sidebar.markdown(
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/snsupratim/) "
        "[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/snsupratim/) "
        "[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/snsupratim/)"
    )

    # ── API key ──────────────────────────────────────────────────────────────
    api_key = st.sidebar.text_input(
        "🔑 Google API Key",
        type="password",
        placeholder="Paste your key here",
    )
    st.sidebar.markdown("Get a key → [ai.google.dev](https://ai.google.dev/)")
    if not api_key:
        st.sidebar.warning("API key required to proceed.")
        st.info("Enter your Google API Key in the sidebar to get started.")
        return

    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.subheader("Menu")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"],              # Fix 10: restrict uploader to PDFs only
        )

        col1, col2 = st.columns(2)
        process_btn = col1.button("✅ Process")
        reset_btn   = col2.button("🔄 Reset")

        if process_btn:
            if pdf_docs:
                with st.spinner("Processing PDFs…"):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks   = get_text_chunks(raw_text)
                    build_vector_store(chunks)
                st.success("PDFs processed! Ask your question below.")
            else:
                st.warning("Upload at least one PDF first.")

        if reset_btn:
            st.session_state.conversation_history = []
            # Fix 11: st.rerun() properly resets widget state;
            #         reassigning local vars had no effect previously.
            st.rerun()

    # ── Main input ───────────────────────────────────────────────────────────
    user_question = st.text_input(
        "💬 Ask a question about your PDFs",
        key="user_question",          # Fix 12: key binds widget to session state
    )

    if user_question:
        handle_question(
            user_question,
            api_key,
            pdf_docs,
            st.session_state.conversation_history,
        )
        st.snow()


if __name__ == "__main__":
    main()