# 🤖 DocuChat — Open Source Edition

> RAG-powered document assistant — 100% local, no API key, no cloud required.

## ✅ Everything is Open Source

| Component | License | Runs Where |
|---|---|---|
| Streamlit (UI) | Apache 2.0 | Local |
| LangChain | MIT | Local |
| FAISS (Vector Store) | MIT (Meta) | Local |
| `all-MiniLM-L6-v2` (Embeddings) | Apache 2.0 | Local |
| Ollama (LLM runner) | MIT | Local |
| Llama 3.2 / Mistral / Gemma (models) | Open weights | Local via Ollama |
| PyPDF2, python-docx | Open Source | Local |

**No Groq. No API key. No cloud. 100% on your machine.**

---

## 🚀 Quick Start

### Step 1 — Install Ollama
Download from **https://ollama.com** and install it.

### Step 2 — Pull a model
```bash
ollama pull llama3.2
```
Other good models you can use:
```bash
ollama pull mistral        # Good all-rounder
ollama pull gemma2         # Google's open model
ollama pull phi3           # Microsoft, small & fast
ollama pull llama3.1:8b    # Meta Llama 3.1
```

### Step 3 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run docuchat/ui/app.py
```
Open **http://localhost:8501**

### Step 5 — Use the app
1. Make sure Ollama is running (`ollama serve` in a terminal)
2. Select your model from the sidebar dropdown
3. Upload a PDF, DOCX, or TXT file
4. Ask questions — get answers from your documents! ✅

---

## 🏗️ Architecture

```
  INDEXING (on upload)
  ─────────────────────────────────────────────────
  PDF/DOCX/TXT → Text Extraction → _clean_text()
                       ↓
          RecursiveCharacterTextSplitter
          chunk_size=1000 | overlap=200
                       ↓
          HuggingFace Embeddings (local)
          all-MiniLM-L6-v2
                       ↓
              FAISS Vector Store (local)

  RETRIEVAL + GENERATION (on question)
  ─────────────────────────────────────────────────
  User Question → Embed → MMR Search (top 6)
                       ↓
          Score Filter (≥ 0.25)
                       ↓
     System Prompt + History + Context + Question
                       ↓
       Ollama Local LLM (llama3.2 / mistral / etc)
                       ↓
          Grounded Answer with Source References ✅
```

---

## 📁 Project Structure

```
Docuchat-opensource/
├── docuchat/
│   ├── core/
│   │   ├── document.py     # PDF/DOCX/TXT text extraction
│   │   ├── rag.py          # FAISS + Ollama RAG pipeline
│   │   └── validator.py    # Ollama connection validator
│   └── ui/
│       └── app.py          # Streamlit chat UI
├── tests/
│   ├── evaluate_rag.py     # Retrieval accuracy evaluation
│   ├── test_unit.py        # Unit tests
│   └── fixtures/           # Sample test documents
├── requirements.txt
└── README.md
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| "Ollama is not running" | Run `ollama serve` in a terminal |
| "Model not found" | Run `ollama pull llama3.2` |
| Slow responses | Use a smaller model: `ollama pull phi3` |
| Scanned PDF has no text | Use OCR first (Tesseract), PyPDF2 can't read image PDFs |
| Port 8501 busy | `streamlit run docuchat/ui/app.py --server.port 8502` |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` |

---

## 🔒 Privacy

- ✅ All documents stay on your machine
- ✅ Embeddings computed locally
- ✅ LLM runs locally via Ollama
- ✅ Zero data sent to any cloud service
- ✅ No API keys needed
