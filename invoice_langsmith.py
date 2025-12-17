import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from pypdf import PdfReader

# LangSmith imports
from langsmith import traceable  # decorator to trace functions


# --- Load environment variables (includes LangSmith settings) ---
load_dotenv()

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/chat"   # Ollama chat endpoint
MODEL_NAME = "llama3.2"                          # Ensure this model has been pulled: `ollama pull llama3.2`


# --- PDF processing ---

@traceable(run_type="tool", name="extract_text_from_pdf", tags=["pdf", "preprocessing"])
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Read all pages from a PDF and return the combined text.
    Traced by LangSmith as a 'tool' run.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    all_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        all_text.append(text)

    return "\n\n".join(all_text).strip()


def build_prompt(pdf_text: str) -> str:
    return f"""
You are an information extraction assistant.

You will be given the text content of a PDF document (such as an invoice, form, or statement).
Your task is to extract the following key fields and return a STRICT JSON object only, with no extra text:

Required keys:
- "invoice_number"
- "invoice_date"
- "customer_name"
- "total_amount"
- "currency"

Rules:
- If a field is missing or not clearly available, set its value to null.
- Do NOT include any additional keys.
- Do NOT add explanations.
- Output must be valid JSON only.

PDF TEXT:
\"\"\"{pdf_text}\"\"\"
"""


# --- LLM (Ollama + Llama 3.2) call wrapped with LangSmith tracing ---

@traceable(run_type="llm", name="ollama_llama3.2_call", tags=["ollama", "llama3.2"])
def call_ollama_llama32(prompt: str) -> dict:
    """
    Call the local Ollama server with Llama 3.2 via the /api/chat endpoint.
    Traced by LangSmith as an LLM run.
    Returns the parsed JSON object from the model's response.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract structured data and respond ONLY with JSON."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()

    # For /api/chat, the text is typically in data["message"]["content"]
    model_text = data.get("message", {}).get("content", "").strip()

    # Try to parse JSON
    try:
        result = json.loads(model_text)
    except json.JSONDecodeError:
        # If the model added extra text, you might need a more robust JSON extraction.
        raise ValueError(f"Model did not return valid JSON. Raw output:\n{model_text}")

    return result


# --- High-level pipeline traced as a "chain" ---

@traceable(run_type="chain", name="pdf_to_key_values_pipeline", tags=["pipeline", "pdf_extraction"])
def extract_key_values_from_pdf(pdf_path: str) -> dict:
    """
    High-level helper: from PDF path to structured key-value dict.
    This is traced as a chain in LangSmith, with nested traces for PDF reading and LLM call.
    """
    pdf_text = extract_text_from_pdf(pdf_path)
    prompt = build_prompt(pdf_text)
    result = call_ollama_llama32(prompt)
    return result


if __name__ == "__main__":
    pdf_file = "invoice.pdf"   # <-- change to your PDF path

    print(f"Using LangSmith project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print(f"Tracing enabled: {os.getenv('LANGSMITH_TRACING', 'false')}")

    try:
        key_values = extract_key_values_from_pdf(pdf_file)
        print("Extracted key values:")
        print(json.dumps(key_values, indent=2, ensure_ascii=False))
        print("\nRun is traced in LangSmith. Open your project to inspect the trace tree.")
    except Exception as e:
        print(f"Error: {e}")