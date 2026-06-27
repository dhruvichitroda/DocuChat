"""Streamlit UI — DocuChat with Groq API (fast) + Ollama (local) support."""

import os
import uuid
from datetime import datetime

import streamlit as st

from docuchat.core import build_vector_store, extract_text_from_file, get_ai_response
from docuchat.core.rag import check_ollama_running, get_available_ollama_models, GROQ_MODELS
from docuchat.core.validator import validate_groq_api_key, validate_ollama_model

st.set_page_config(page_title="DocuChat", page_icon="🤖", layout="wide")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Session state
for key, default in [
    ("files", []), ("conversation", []), ("known_files", set()),
    ("vector_store", None), ("groq_api_key", ""), ("selected_mode", "Groq API (Fast ⚡)"),
    ("selected_groq_model", "llama-3.3-70b-versatile"), ("selected_ollama_model", "gemma2:latest"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _rebuild_vector_store():
    st.session_state.vector_store = build_vector_store(st.session_state.files) if st.session_state.files else None


def _remove_file(file_id, original_name, size_bytes, path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    st.session_state.files = [f for f in st.session_state.files if f.get("id") != file_id]
    st.session_state.known_files.discard(f"{original_name}:{size_bytes}")
    _rebuild_vector_store()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 DocuChat")

    # Mode selector
    st.subheader("⚙️ LLM Mode")
    mode = st.radio(
        "Choose speed vs privacy:",
        ["Groq API (Fast ⚡)", "Ollama (Local 🔒)"],
        index=0,
    )
    st.session_state.selected_mode = mode

    st.divider()

    # ── Groq mode ──
    if mode == "Groq API (Fast ⚡)":
        st.subheader("🔑 Groq API Key")
        st.markdown("Free key → [console.groq.com/keys](https://console.groq.com/keys)")
        api_key_input = st.text_input("Paste your key (gsk_...)", value=st.session_state.groq_api_key, type="password")
        if api_key_input != st.session_state.groq_api_key:
            st.session_state.groq_api_key = api_key_input.strip()

        if st.session_state.groq_api_key:
            is_valid, msg = validate_groq_api_key(st.session_state.groq_api_key)
            if is_valid:
                st.success("✅ Valid key")
            else:
                st.error(f"❌ {msg}")

        st.subheader("🤖 Groq Model")
        groq_model = st.selectbox("Select model", GROQ_MODELS, index=0)
        st.session_state.selected_groq_model = groq_model
        st.caption("llama-3.3-70b = most accurate | llama-3.1-8b = fastest")

    # ── Ollama mode ──
    else:
        st.subheader("🦙 Ollama (Local)")
        ollama_ok, ollama_msg = check_ollama_running()
        if ollama_ok:
            st.success("✅ Ollama is running")
        else:
            st.error(f"❌ {ollama_msg}")
            st.code("ollama serve")

        available_models = get_available_ollama_models()
        ollama_model = st.selectbox("Select model", available_models, index=0)
        st.session_state.selected_ollama_model = ollama_model

        if ollama_ok:
            is_valid, val_msg = validate_ollama_model(ollama_model)
            if is_valid:
                st.success(f"✅ Model ready: `{ollama_model}`")
            else:
                st.warning(f"⚠️ {val_msg}")

    st.divider()

    # ── File upload ──
    st.subheader("📄 Documents")
    uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded:
        new_files_added = False
        for file in uploaded:
            try:
                size_bytes = getattr(file, "size", None) or len(file.getbuffer())
                unique_key = f"{file.name}:{size_bytes}"
                if unique_key in st.session_state.known_files:
                    continue
                file_id = f"{uuid.uuid4()}_{file.name}"
                file_path = os.path.join(UPLOAD_DIR, file_id)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                st.session_state.files.append({
                    "id": file_id, "original_name": file.name, "path": file_path,
                    "size": os.path.getsize(file_path),
                    "text_content": extract_text_from_file(file_path, file.name),
                    "uploaded_at": datetime.now().isoformat(),
                })
                st.session_state.known_files.add(unique_key)
                st.toast(f"✅ Uploaded {file.name}")
                new_files_added = True
            except Exception as e:
                st.warning(f"Failed to process {file.name}: {e}")
        if new_files_added:
            with st.spinner("Building knowledge base…"):
                _rebuild_vector_store()

    if st.session_state.files:
        for f in list(st.session_state.files):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.write(f"📄 {f['original_name']} ({round(f['size']/1024,1)} KB)")
            with col2:
                if st.button("🗑", key=f"rm_{f['id']}", use_container_width=True):
                    _remove_file(f["id"], f["original_name"], int(f.get("size",0)), f.get("path",""))
                    st.rerun()

    st.divider()
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.conversation = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🤖 DocuChat")

mode = st.session_state.selected_mode
is_groq = mode == "Groq API (Fast ⚡)"
groq_key_valid = validate_groq_api_key(st.session_state.groq_api_key)[0] if is_groq else False

# Status hints
if not st.session_state.files or (is_groq and not groq_key_valid):
    col1, col2, col3 = st.columns(3)
    with col1:
        if is_groq:
            status = "✅" if groq_key_valid else "❌"
            st.info(f"{status} **Step 1** — Paste your free Groq API key in sidebar")
        else:
            ollama_ok, _ = check_ollama_running()
            status = "✅" if ollama_ok else "❌"
            st.info(f"{status} **Step 1** — Start Ollama (`ollama serve`)")
    with col2:
        if is_groq:
            st.info("⚡ **Step 2** — Select Groq model in sidebar")
        else:
            st.info("🦙 **Step 2** — Select Ollama model in sidebar")
    with col3:
        status = "✅" if st.session_state.files else "⏳"
        st.info(f"{status} **Step 3** — Upload your document")

# Speed badge
if is_groq:
    st.caption("⚡ Groq API mode — fast cloud responses | model: " + st.session_state.selected_groq_model)
else:
    st.caption("🔒 Ollama local mode — private, no internet needed | model: " + st.session_state.selected_ollama_model)

# Chat history
for msg in st.session_state.conversation:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])


def handle_user_message(user_message: str):
    if not user_message:
        return
    if not st.session_state.files:
        st.warning("⚠️ Please upload a document first.")
        return

    is_groq = st.session_state.selected_mode == "Groq API (Fast ⚡)"

    if is_groq:
        is_valid, msg = validate_groq_api_key(st.session_state.groq_api_key)
        if not is_valid:
            st.warning(f"⚠️ {msg} — Get free key at console.groq.com/keys")
            return
        model = st.session_state.selected_groq_model
        groq_key = st.session_state.groq_api_key
    else:
        ollama_ok, err = check_ollama_running()
        if not ollama_ok:
            st.warning(f"⚠️ {err}")
            return
        model = st.session_state.selected_ollama_model
        groq_key = ""

    if not st.session_state.vector_store:
        with st.spinner("Building knowledge base…"):
            _rebuild_vector_store()
        if not st.session_state.vector_store:
            st.error("Could not build knowledge base.")
            return

    st.session_state.conversation.append({"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()})

    with st.chat_message("assistant"):
        spinner_msg = "Thinking with Groq ⚡…" if is_groq else f"Thinking with {model}…"
        with st.spinner(spinner_msg):
            answer = get_ai_response(
                question=user_message,
                vector_store=st.session_state.vector_store,
                model_name=model,
                conversation_history=st.session_state.conversation,
                groq_api_key=groq_key,
            )
        st.markdown(answer)

    st.session_state.conversation.append({"role": "assistant", "content": answer, "timestamp": datetime.now().isoformat()})


if prompt := st.chat_input("Ask a question about your documents…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    handle_user_message(prompt)
