from __future__ import annotations

from pathlib import Path

import streamlit as st

from ollama_client import OLLAMA_MODEL, generate_with_ollama, is_ollama_enabled
from rag import TransportationRAG


APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = APP_DIR / "data" / "transport_docs"


def load_rag() -> TransportationRAG:
    return TransportationRAG(DOCS_DIR)


def answer_question(rag: TransportationRAG, question: str) -> tuple[str, list]:
    retrieved = rag.retrieve(question, top_k=3)
    prompt = rag.build_prompt(question, retrieved)
    answer = generate_with_ollama(prompt)

    if not answer:
        answer = rag.retrieval_only_answer(question, retrieved)

    return answer, retrieved


st.set_page_config(page_title="Transportation RAG Chatbot", page_icon="", layout="centered")

# Simplified, safer CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    /* Force dark text for visibility */
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label {
        color: #1e293b !important;
    }
    .stApp [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] h2 {
        color: #3b82f6 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Transportation AI")
st.caption("Intelligent Logistics & Transit Assistant")

import time

with st.sidebar:
    st.header("🗂️ Knowledge Base")
    # Ensure directory exists before listing
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    docs = sorted(path.name for path in DOCS_DIR.glob("*.txt"))
    st.write(f"**{len(docs)}** resources indexed")
    
    for doc in docs:
        st.info(f"📄 {doc}")

    st.markdown("---")
    st.write("🟢 **Assistant Online**")
    st.caption("v1.2.1 - Enhanced Search")

rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your Transportation AI assistant. I can help you with "
                "schedules, routes, and shipping logistics. How can I assist you today?"
            ),
            "sources": [],
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source.source}** - relevance {source.score:.2f}\n\n"
                        f"{source.text[:500]}..."
                    )

question = st.chat_input("Ask a transportation question")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("AI is analyzing local transport data..."):
            # Simulate "AI Thinking Time"
            time.sleep(1.2)
            answer, sources = answer_question(rag, question)
        st.markdown(answer)
        if sources:
            with st.expander("References"):
                for source in sources:
                    st.markdown(
                        f"**{source.source}**\n\n"
                        f"{source.text[:300]}..."
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
