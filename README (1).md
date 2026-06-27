# 🤖 DocuChat — AI-Powered Document Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=for-the-badge&logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-API-purple?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)

**Ask questions about your documents using AI — Fast, Private, and 100% Free**

</div>

---

## 📌 About The Project

DocuChat is a **RAG (Retrieval-Augmented Generation)** powered document assistant that lets you upload any PDF, DOCX, or TXT file and ask questions about it in natural language.

It supports two modes:
- ⚡ **Groq API Mode** — Ultra-fast responses using cloud LLMs (Free API)
- 🔒 **Ollama Local Mode** — 100% private, runs entirely on your machine

---

## ✨ Features

- 📄 Upload **PDF, DOCX, TXT** files
- ⚡ **Groq API** integration — Llama 3.3 70B, Mixtral models
- 🔒 **Ollama** local LLM support — Gemma2, Mistral, Phi3
- 🧠 **RAG Pipeline** — FAISS vector store + semantic search
- 💬 **Conversation Memory** — remembers last 3 chat turns
- 📚 **Multi-document** support — upload and query multiple files
- 🎯 **Source citations** in answers — [Source: filename]
- 🔍 **MMR Search** — diverse and relevant chunk retrieval

---

## 🏗️ Architecture

```
  INDEXING (on upload)
  ──────────────────────────────────────
  PDF/DOCX/TXT → Text Extraction
                      ↓
         RecursiveCharacterTextSplitter
         chunk_size=1000 | overlap=200
                      ↓
         HuggingFace Embeddings (local)
         all-MiniLM-L6-v2
                      ↓
             FAISS Vector Store

  RETRIEVAL + GENERATION (on question)
  ──────────────────────────────────────
  Question → Embed → MMR Search (top 6)
                      ↓
          Score Filter (≥ 0.25)
                      ↓
    System Prompt + History + Context
                      ↓
       Groq API ⚡ OR Ollama Local 🔒
                      ↓
      Grounded Answer + Source Reference ✅
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **UI Framework** | Streamlit |
| **LLM Framework** | LangChain |
| **Cloud LLM** | Groq API (Llama 3.3 70B, Mixtral) |
| **Local LLM** | Ollama (Gemma2, Mistral, Phi3) |
| **Vector Store** | FAISS (by Meta) |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 |
| **Doc Parsing** | PyPDF2, python-docx |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Ollama (for local mode) — [ollama.com](https://ollama.com)
- Groq API Key (for fast mode) — [console.groq.com/keys](https://console.groq.com/keys) (Free)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/dhruvichitroda/DocuChat.git
cd DocuChat
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set Python path**
```bash
# Windows
set PYTHONPATH=%cd%

# Mac/Linux
export PYTHONPATH=$(pwd)
```

**4. Run the app**
```bash
streamlit run docuchat/ui/app.py
```

**5. Open browser**
```
http://localhost:8501
```

---

## 💡 How To Use

### ⚡ Groq API Mode (Fast)
1. Get free API key from [console.groq.com/keys](https://console.groq.com/keys)
2. Select **"Groq API (Fast ⚡)"** in sidebar
3. Paste your `gsk_...` API key
4. Select model — `llama-3.3-70b-versatile` (recommended)
5. Upload your document
6. Ask questions and get answers in 2-5 seconds!

### 🔒 Ollama Local Mode (Private)
```bash
# Pull a model
ollama pull gemma2

# Start Ollama
ollama serve
```
1. Select **"Ollama (Local 🔒)"** in sidebar
2. Select your model
3. Upload document and start chatting!

---

## 📁 Project Structure

```
DocuChat/
├── docuchat/
│   ├── core/
│   │   ├── document.py     # PDF/DOCX/TXT text extraction
│   │   ├── rag.py          # FAISS + RAG pipeline
│   │   └── validator.py    # API key & model validation
│   └── ui/
│       └── app.py          # Streamlit chat interface
├── tests/
│   ├── evaluate_rag.py     # RAG evaluation (96.7% Hit Rate)
│   ├── test_unit.py        # Unit tests
│   └── fixtures/           # Sample test documents
├── requirements.txt
└── README.md
```

---

## 🎯 Use Cases

- 📝 **Resume Analysis** — Extract skills, experience, summarize
- 📜 **Legal Documents** — Understand contracts and clauses
- 🔬 **Research Papers** — Get key findings instantly
- 📊 **Business Reports** — Summarize and query data
- 📚 **Study Material** — Ask questions from textbooks

---

## 👩‍💻 Developer

**Dhruvi Chitroda**

[![GitHub](https://img.shields.io/badge/GitHub-dhruvichitroda-black?style=flat&logo=github)](https://github.com/dhruvichitroda)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
