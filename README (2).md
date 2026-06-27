# 🤖 DocuChat — AI-Powered Document Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-API%20⚡-F55036?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM%20🔒-000000?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### Upload any document. Ask any question. Get instant AI answers.

⚡ **Groq API Mode** — 2-5 second responses using Llama 3.3 70B (Free)

🔒 **Ollama Local Mode** — 100% private, runs entirely on your machine

</div>

---

## 📌 What is DocuChat?

DocuChat is a **RAG (Retrieval-Augmented Generation)** powered document Q&A assistant. Instead of reading through entire documents manually, simply upload your file and ask questions in plain English — DocuChat finds the exact answer from your document using AI.

**Built for:** Resumes, Legal Documents, Research Papers, Business Reports, Study Material

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Multi-format Support | PDF, DOCX, TXT file upload |
| ⚡ Groq API Mode | Ultra-fast responses — Llama 3.3 70B, Mixtral (Free API) |
| 🔒 Ollama Local Mode | 100% offline — Gemma2, Mistral, Phi3 |
| 🧠 Smart RAG Pipeline | FAISS vector store + MMR semantic search |
| 💬 Conversation Memory | Remembers last 3 chat turns for follow-ups |
| 📚 Multi-document | Upload and query multiple files at once |
| 🎯 Source Citations | Every answer references its source document |
| 🔍 MMR Retrieval | Diverse and accurate chunk selection |

---

## 🏗️ System Architecture

```
  ┌─────────────────────────────────────────────┐
  │           INDEXING  (on file upload)         │
  ├─────────────────────────────────────────────┤
  │                                             │
  │  PDF / DOCX / TXT                           │
  │       │                                     │
  │       ▼                                     │
  │  Text Extraction + Cleaning                 │
  │       │                                     │
  │       ▼                                     │
  │  RecursiveCharacterTextSplitter             │
  │  chunk_size=1000  |  overlap=200            │
  │       │                                     │
  │       ▼                                     │
  │  HuggingFace Embeddings (Local)             │
  │  Model: all-MiniLM-L6-v2                   │
  │       │                                     │
  │       ▼                                     │
  │  FAISS Vector Store  ✅                     │
  └─────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────┐
  │      RETRIEVAL + GENERATION (on question)   │
  ├─────────────────────────────────────────────┤
  │                                             │
  │  User Question                              │
  │       │                                     │
  │       ▼                                     │
  │  Embed Question → MMR Search (Top 6 Chunks) │
  │       │                                     │
  │       ▼                                     │
  │  Score Filter (threshold ≥ 0.25)            │
  │       │                                     │
  │       ▼                                     │
  │  System Prompt + History + Context          │
  │       │                                     │
  │       ├──────────────┬──────────────────┐   │
  │       ▼              ▼                  │   │
  │  Groq API ⚡    Ollama Local 🔒         │   │
  │  (Fast Cloud)   (Private Local)         │   │
  │       │              │                  │   │
  │       └──────────────┘                  │   │
  │                 │                       │   │
  │                 ▼                       │   │
  │   Grounded Answer + [Source: file] ✅   │   │
  └─────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit | Interactive chat interface |
| **LLM Framework** | LangChain | Orchestration & prompt management |
| **Cloud LLM** | Groq API | Fast inference — Llama 3.3 70B, Mixtral |
| **Local LLM** | Ollama | Private local inference — Gemma2, Mistral |
| **Vector Store** | FAISS (Meta) | Fast similarity search |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 | Local text embeddings |
| **PDF Parser** | PyPDF2 | Extract text from PDFs |
| **DOCX Parser** | python-docx | Extract text from Word files |
| **Language** | Python 3.11+ | Core language |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Git

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
# Windows (Command Prompt)
set PYTHONPATH=%cd%

# Mac / Linux
export PYTHONPATH=$(pwd)
```

**4. Run the app**
```bash
streamlit run docuchat/ui/app.py
```

**5. Open in browser**
```
http://localhost:8501
```

---

## 💡 Usage Guide

### ⚡ Mode 1 — Groq API (Fast, Recommended)

> Free API key, no credit card needed, 2-5 second responses

1. Get free key from 👉 [console.groq.com/keys](https://console.groq.com/keys)
2. In sidebar → select **"Groq API (Fast ⚡)"**
3. Paste your `gsk_...` key
4. Select model → `llama-3.3-70b-versatile` *(most accurate)*
5. Upload your document
6. Ask questions → get answers in seconds!

**Available Groq Models:**
| Model | Speed | Best For |
|---|---|---|
| `llama-3.3-70b-versatile` | Fast | Most accurate answers |
| `llama-3.1-8b-instant` | Very Fast | Quick summaries |
| `mixtral-8x7b-32768` | Fast | Long documents |

---

### 🔒 Mode 2 — Ollama (Local, Private)

> No internet required, data never leaves your machine

```bash
# Step 1: Install Ollama from https://ollama.com

# Step 2: Pull a model
ollama pull gemma2

# Step 3: Start Ollama server
ollama serve
```

1. In sidebar → select **"Ollama (Local 🔒)"**
2. Select your downloaded model
3. Upload document and chat!

**Available Ollama Models:**
| Model | Size | Best For |
|---|---|---|
| `gemma2:latest` | 5.4 GB | Best quality local |
| `gemma:2b` | 1.7 GB | Fast, lightweight |
| `mistral` | 4 GB | Good all-rounder |
| `phi3` | 2 GB | Small & fast |

---

## 🎯 Example Questions

**For a Resume:**
```
Summarize this resume in 5 bullet points
What are the technical skills of this candidate?
Is this candidate suitable for a Python developer role?
What is the candidate's educational background?
What improvements can be made to this resume?
```

**For any Document:**
```
What is this document about?
Summarize the main points
What does it say about [topic]?
List all key findings
```

---

## 📁 Project Structure

```
DocuChat/
├── docuchat/
│   ├── core/
│   │   ├── __init__.py       # Core module exports
│   │   ├── document.py       # PDF / DOCX / TXT text extraction
│   │   ├── rag.py            # FAISS + RAG pipeline (Groq + Ollama)
│   │   └── validator.py      # API key & model validation
│   └── ui/
│       ├── __init__.py
│       └── app.py            # Streamlit chat interface
├── tests/
│   ├── evaluate_rag.py       # RAG evaluation — 96.7% Hit Rate @6
│   ├── test_unit.py          # Unit tests
│   └── fixtures/             # Sample test documents
│       ├── company_policy.txt
│       ├── product_spec.txt
│       └── research_paper.txt
├── results/
│   └── eval_report.json      # RAG evaluation results
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project config
├── run.bat                   # Windows one-click launcher
├── run.sh                    # Mac/Linux one-click launcher
└── README.md
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: docuchat` | Run `set PYTHONPATH=%cd%` then restart app |
| `Ollama not running` | Run `ollama serve` in a new terminal |
| `Model not found` | Run `ollama pull gemma2` |
| `Model decommissioned` | Select `llama-3.3-70b-versatile` from dropdown |
| `Invalid Groq API key` | Get free key at console.groq.com/keys |
| `Slow first response` | Normal — embedding model downloads once (~90MB) |
| Port 8501 busy | Add `--server.port 8502` to streamlit command |

---

## 👩‍💻 Developer

**Dhruvi Chitroda**

[![GitHub](https://img.shields.io/badge/GitHub-dhruvichitroda-181717?style=for-the-badge&logo=github)](https://github.com/dhruvichitroda)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
Made with ❤️ using LangChain, FAISS, Groq & Ollama
</div>
