# llm.py
from typing import List, Dict


def fake_llm(messages: List[Dict[str, str]]) -> str:
    """
    Placeholder. Replace with actual API call (OpenAI, HF Inference, etc.)
    For now we just echo last user message with a prefix.
    """
    last_user = next(m for m in reversed(messages) if m["role"] == "user")
    return f"(fake LLM) You said: {last_user['content']}"
