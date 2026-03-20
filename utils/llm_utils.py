"""LLM interaction utilities.

Provides a unified ``run_llm`` interface that routes requests to either a
local Ollama instance or an OpenAI-compatible API, along with small helper
functions for parsing LLM outputs.
"""

import os
import re
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration -- override via environment variables
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")


# ---------------------------------------------------------------------------
# Core LLM caller
# ---------------------------------------------------------------------------


def run_llm(prompt, temperature, max_tokens, openai_api_key=None, engine="gpt-3.5-turbo"):
    """Call an LLM and return the generated text.

    Routing logic:
    - If *engine* contains ``"gpt"`` or ``"gemini"`` (case-insensitive), the
      request is sent to the OpenAI-compatible API specified by
      ``OPENAI_BASE_URL``.
    - Otherwise, the request is sent to a local Ollama server.

    Args:
        prompt: The prompt string.
        temperature: Sampling temperature.
        max_tokens: Maximum number of tokens to generate.
        openai_api_key: API key (falls back to ``OPENAI_API_KEY`` env var).
        engine: Model name / identifier.

    Returns:
        The generated text as a string.

    Raises:
        Exception: After exhausting retry attempts.
    """
    api_key = openai_api_key or OPENAI_API_KEY
    max_retries = 5
    retry_count = 0
    result = None

    # --- Local Ollama models ---
    if "gpt" not in engine.lower() and "gemini" not in engine.lower():
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": engine,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        while result is None and retry_count < max_retries:
            try:
                response = requests.post(OLLAMA_URL, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json().get("response", "")
            except requests.exceptions.RequestException as e:
                print(f"{engine} error, retry {retry_count + 1}/{max_retries}: {e}")
                retry_count += 1
                time.sleep(2)
        if result is None:
            raise Exception(f"Failed to connect to {engine} after {max_retries} retries")

    # --- OpenAI-compatible API ---
    else:
        client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
        messages = [{"role": "user", "content": prompt}]
        while result is None and retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                result = response.choices[0].message.content
            except Exception as e:
                print(f"{engine} error, retry {retry_count + 1}/{max_retries}: {e}")
                retry_count += 1
                time.sleep(2)
        if result is None:
            raise Exception(f"Failed to connect to {engine} after {max_retries} retries")

    return result


# ---------------------------------------------------------------------------
# Output parsing helpers
# ---------------------------------------------------------------------------


def extract_answer(text):
    """Extract the first ``{...}`` delimited answer from *text*."""
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index + 1 : end_index].strip()
    return ""


def if_true(prompt):
    """Return ``True`` if *prompt* is a case-insensitive ``'yes'``."""
    return prompt.lower().strip().replace(" ", "") == "yes"


def clean_scores(string, entity_candidates):
    """Parse float scores from *string*, one per entity candidate.

    If the number of parsed scores does not match the candidate list length,
    returns uniform scores.
    """
    scores = re.findall(r"\d+\.\d+", string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    print("Score count mismatch; assigning uniform scores.")
    return [1 / len(entity_candidates)] * len(entity_candidates)
