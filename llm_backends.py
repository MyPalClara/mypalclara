from __future__ import annotations

import os
from typing import Callable, List, Dict

from openai import OpenAI
from huggingface_hub import InferenceClient


def make_llm() -> Callable[[List[Dict[str, str]]], str]:
    """
    Return a function(messages) -> assistant_reply string.

    Select backend with env var LLM_PROVIDER:
      - "openrouter" (default)
      - "huggingface"
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider == "openrouter":
        return _make_openrouter_llm()
    elif provider == "huggingface":
        return _make_hf_llm()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER={provider}")


def _make_openrouter_llm() -> Callable[[List[Dict[str, str]]], str]:
    """
    Uses OpenRouter via OpenAI-compatible client.

    Required env vars:
      - OPENROUTER_API_KEY
      - OPENROUTER_MODEL
      - OPENROUTER_SITE
      - OPENROUTER_TITLE
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    site = os.getenv("OPENROUTER_SITE", "http://localhost:7860")
    title = os.getenv("OPENROUTER_TITLE", "Clara Assistant")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": site,
            "X-Title": title,
        },
    )

    def llm(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content

    return llm


def _make_hf_llm() -> Callable[[List[Dict[str, str]]], str]:
    """
    Uses HuggingFace Inference API via InferenceClient.chat_completion.

    Required env vars:
      - HF_API_TOKEN
      - HF_MODEL
    """
    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_API_TOKEN is not set")

    model = os.getenv(
        "HF_MODEL",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    )

    client = InferenceClient(token=token, model=model)

    def llm(messages: List[Dict[str, str]]) -> str:
        resp = client.chat_completion(
            messages=messages, max_tokens=512, stream=False)
        return resp.choices[0].message.content

    return llm
