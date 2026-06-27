#!/bin/bash
echo "Starting DocuChat (Open Source)..."
echo "Make sure Ollama is running: ollama serve"
echo ""
pip install -r requirements.txt -q
streamlit run docuchat/ui/app.py
