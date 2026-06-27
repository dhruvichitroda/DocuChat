"""RAG pipeline — supports both Groq API (fast) and Ollama (local)."""

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

@st.cache_resource(show_spinner="Loading embedding model…")
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200
_TOP_K = 6
_FETCH_K = 20
_SCORE_THRESHOLD = 0.25
_MAX_HISTORY = 3

# Groq models (fast, free API)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

_SYSTEM_PROMPT = (
    "You are an expert document analyst. Answer the user's question STRICTLY "
    "based on the provided document context.\n\n"
    "Rules:\n"
    "1. Only use information from the context. Do NOT rely on outside knowledge.\n"
    "2. If the answer is not in the context, say: "
    "'I could not find this information in the uploaded documents.'\n"
    "3. Reference the source document when relevant (e.g. [Source: filename]).\n"
    "4. Be accurate, concise, and well-structured.\n"
    "5. For follow-up questions, use the conversation history to understand context."
)


def get_available_ollama_models() -> list[str]:
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]
            models = [line.split()[0] for line in lines if line.strip()]
            return models if models else ["gemma2:latest"]
    except Exception:
        pass
    return ["gemma2:latest"]


def check_ollama_running() -> tuple[bool, str]:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=3)
        return True, "Ollama is running"
    except Exception:
        return False, "Ollama is not running. Start it with: ollama serve"


def build_vector_store(files: list[dict]) -> FAISS | None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    docs = []
    for file in files:
        content = file.get("text_content", "").strip()
        if not content:
            continue
        chunks = splitter.create_documents(
            [content],
            metadatas=[{"source": file["original_name"]}],
        )
        docs.extend(chunks)
    return FAISS.from_documents(docs, _get_embeddings()) if docs else None


def _build_messages(context: str, question: str, conversation_history):
    messages: list = [SystemMessage(content=_SYSTEM_PROMPT)]
    if conversation_history:
        for turn in conversation_history[-(_MAX_HISTORY * 2):]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=f"Document Context:\n{context}\n\nQuestion: {question}"))
    return messages


def get_ai_response(
    question: str,
    vector_store: FAISS,
    model_name: str = "gemma2:latest",
    conversation_history: list[dict] | None = None,
    groq_api_key: str = "",
) -> str:
    try:
        # Retrieve relevant chunks
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": _TOP_K, "fetch_k": _FETCH_K, "lambda_mult": 0.7},
        )
        mmr_docs = retriever.invoke(question)
        scored = vector_store.similarity_search_with_relevance_scores(question, k=_TOP_K)
        good_docs = [doc for doc, score in scored if score >= _SCORE_THRESHOLD]
        final_docs = good_docs if good_docs else mmr_docs

        context_parts = []
        for i, doc in enumerate(final_docs, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        messages = _build_messages(context, question, conversation_history)

        # Use Groq if API key provided, else Ollama
        if groq_api_key and groq_api_key.startswith("gsk_"):
            from langchain_groq import ChatGroq
            llm = ChatGroq(api_key=groq_api_key, model_name=model_name, max_tokens=2048, temperature=0.1)
        else:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=model_name, temperature=0.1)

        return llm.invoke(messages).content

    except Exception as e:
        error = str(e)
        if "401" in error or "authentication" in error.lower() or "invalid api key" in error.lower():
            return "❌ Invalid Groq API key. Please check your key at console.groq.com/keys"
        if "connection refused" in error.lower() or "connect" in error.lower():
            return "❌ Cannot connect to Ollama. Run `ollama serve` in a terminal."
        if "model" in error.lower() and "not found" in error.lower():
            return f"❌ Model `{model_name}` not found. Run `ollama pull {model_name}`"
        return f"Error: {error}"
