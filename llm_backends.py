from __future__ import annotations

import os
from typing import Callable, List, Dict, Generator

from openai import OpenAI

# Global clients for reuse
_openrouter_client: OpenAI = None
_nanogpt_client: OpenAI = None
_custom_openai_client: OpenAI = None


def _get_openrouter_client() -> OpenAI:
    """Get or create OpenRouter client."""
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        site = os.getenv("OPENROUTER_SITE", "http://localhost:3000")
        title = os.getenv("OPENROUTER_TITLE", "MyPalClara")

        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": site,
                "X-Title": title,
            },
        )
    return _openrouter_client


def _get_nanogpt_client() -> OpenAI:
    """Get or create NanoGPT client."""
    global _nanogpt_client
    if _nanogpt_client is None:
        api_key = os.getenv("NANOGPT_API_KEY")
        if not api_key:
            raise RuntimeError("NANOGPT_API_KEY is not set")

        _nanogpt_client = OpenAI(
            base_url="https://nano-gpt.com/api/v1",
            api_key=api_key,
        )
    return _nanogpt_client


def _get_custom_openai_client() -> OpenAI:
    """Get or create custom OpenAI-compatible client."""
    global _custom_openai_client
    if _custom_openai_client is None:
        api_key = os.getenv("CUSTOM_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("CUSTOM_OPENAI_API_KEY is not set")

        base_url = os.getenv("CUSTOM_OPENAI_BASE_URL", "https://api.openai.com/v1")

        _custom_openai_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    return _custom_openai_client


def make_llm() -> Callable[[List[Dict[str, str]]], str]:
    """
    Return a function(messages) -> assistant_reply string.

    Select backend with env var LLM_PROVIDER:
      - "openrouter" (default)
      - "nanogpt"
      - "openai" (custom OpenAI-compatible endpoint)
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider == "openrouter":
        return _make_openrouter_llm()
    elif provider == "nanogpt":
        return _make_nanogpt_llm()
    elif provider == "openai":
        return _make_custom_openai_llm()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER={provider}")


def _make_openrouter_llm() -> Callable[[List[Dict[str, str]]], str]:
    """Non-streaming OpenRouter LLM."""
    client = _get_openrouter_client()
    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")

    def llm(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content

    return llm


def _make_nanogpt_llm() -> Callable[[List[Dict[str, str]]], str]:
    """Non-streaming NanoGPT LLM."""
    client = _get_nanogpt_client()
    model = os.getenv("NANOGPT_MODEL", "moonshotai/Kimi-K2-Instruct-0905")

    def llm(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content

    return llm


def _make_custom_openai_llm() -> Callable[[List[Dict[str, str]]], str]:
    """Non-streaming custom OpenAI-compatible LLM."""
    client = _get_custom_openai_client()
    model = os.getenv("CUSTOM_OPENAI_MODEL", "gpt-4o")

    def llm(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content

    return llm


def make_llm_streaming() -> Callable[[List[Dict[str, str]]], Generator[str, None, None]]:
    """Return a streaming LLM function that yields chunks."""
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider == "openrouter":
        return _make_openrouter_llm_streaming()
    elif provider == "nanogpt":
        return _make_nanogpt_llm_streaming()
    elif provider == "openai":
        return _make_custom_openai_llm_streaming()
    else:
        raise ValueError(f"Streaming not supported for LLM_PROVIDER={provider}")


def _make_openrouter_llm_streaming() -> Callable[[List[Dict[str, str]]], Generator[str, None, None]]:
    """Streaming OpenRouter LLM."""
    client = _get_openrouter_client()
    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")

    def llm(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return llm


def _make_nanogpt_llm_streaming() -> Callable[[List[Dict[str, str]]], Generator[str, None, None]]:
    """Streaming NanoGPT LLM."""
    client = _get_nanogpt_client()
    model = os.getenv("NANOGPT_MODEL", "moonshotai/Kimi-K2-Instruct-0905")

    def llm(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return llm


def _make_custom_openai_llm_streaming() -> Callable[[List[Dict[str, str]]], Generator[str, None, None]]:
    """Streaming custom OpenAI-compatible LLM."""
    client = _get_custom_openai_client()
    model = os.getenv("CUSTOM_OPENAI_MODEL", "gpt-4o")

    def llm(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return llm
