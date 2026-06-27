"""Validates Groq API key or Ollama model availability."""
import re


def validate_groq_api_key(api_key: str) -> tuple[bool, str]:
    if not api_key or not isinstance(api_key, str):
        return False, "API key is required"
    api_key = api_key.strip()
    if not api_key.startswith("gsk_"):
        return False, "Invalid key format. Groq keys start with 'gsk_'"
    if len(api_key) < 30:
        return False, "API key too short"
    if not re.match(r"^gsk_[A-Za-z0-9_]+$", api_key):
        return False, "Invalid characters in API key"
    return True, "Valid Groq API key"


def validate_ollama_model(model_name: str) -> tuple[bool, str]:
    if not model_name:
        return False, "Model name is required"
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=3)
    except Exception:
        return False, "Ollama is not running. Start with: ollama serve"
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            base_name = model_name.split(":")[0]
            if base_name in result.stdout or model_name in result.stdout:
                return True, f"Model '{model_name}' is ready"
            return False, f"Model '{model_name}' not found. Run: ollama pull {model_name}"
    except Exception as e:
        return False, f"Could not check model: {e}"
    return True, "Model available"
