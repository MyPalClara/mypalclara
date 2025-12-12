from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from mem0 import Memory

load_dotenv()

# Mem0 has its own independent provider config (separate from chat LLM)
MEM0_PROVIDER = os.getenv("MEM0_PROVIDER", "openrouter").lower()
MEM0_MODEL = os.getenv("MEM0_MODEL", "openai/gpt-4o-mini")

# Optional overrides - if not set, uses the provider's default key/url
MEM0_API_KEY = os.getenv("MEM0_API_KEY")
MEM0_BASE_URL = os.getenv("MEM0_BASE_URL")

# OpenAI API for embeddings (always required)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Provider defaults
PROVIDER_DEFAULTS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "nanogpt": {
        "base_url": "https://nano-gpt.com/api/v1",
        "api_key_env": "NANOGPT_API_KEY",
    },
    "openai": {
        "base_url": os.getenv("CUSTOM_OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key_env": "CUSTOM_OPENAI_API_KEY",
    },
}

# IMPORTANT: mem0 auto-detects these env vars and overrides our config!
# We must save and clear them before mem0 initialization, then restore after.
_saved_env_vars = {}
_env_vars_to_clear = ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MEM0_API_KEY"]

def _clear_mem0_env_vars():
    """Clear env vars that mem0 auto-detects, save them for later restoration."""
    for var in _env_vars_to_clear:
        if var in os.environ:
            _saved_env_vars[var] = os.environ.pop(var)
            print(f"[mem0] Temporarily cleared {var} to prevent auto-detection")

def _restore_env_vars():
    """Restore cleared env vars after mem0 initialization."""
    for var, value in _saved_env_vars.items():
        os.environ[var] = value
        print(f"[mem0] Restored {var}")


# Store mem0 data in a local directory
BASE_DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).parent)))
QDRANT_DATA_DIR = BASE_DATA_DIR / "qdrant_data"
QDRANT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_llm_config() -> dict | None:
    """
    Build mem0 LLM config based on MEM0_PROVIDER.

    This is completely independent from the chat LLM provider.
    """
    if MEM0_PROVIDER not in PROVIDER_DEFAULTS:
        print(f"[mem0] Unknown MEM0_PROVIDER={MEM0_PROVIDER} - mem0 LLM disabled")
        return None

    provider_config = PROVIDER_DEFAULTS[MEM0_PROVIDER]

    # Get API key: explicit MEM0_API_KEY > provider's default key
    api_key = MEM0_API_KEY or os.getenv(provider_config["api_key_env"])
    if not api_key:
        print(f"[mem0] No API key found for MEM0_PROVIDER={MEM0_PROVIDER} - mem0 LLM disabled")
        return None

    # Get base URL: explicit MEM0_BASE_URL > provider's default URL
    base_url = MEM0_BASE_URL or provider_config["base_url"]

    print(f"[mem0] Provider: {MEM0_PROVIDER}")
    print(f"[mem0] Model: {MEM0_MODEL}")
    print(f"[mem0] Base URL: {base_url}")

    return {
        "provider": "openai",
        "config": {
            "model": MEM0_MODEL,
            "api_key": api_key,
            "openai_base_url": base_url,
            "temperature": 0,
        },
    }


# Get LLM config
llm_config = _get_llm_config()

# Build config - embeddings always use OpenAI
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mypalclara_memories",
            "path": str(QDRANT_DATA_DIR),
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": OPENAI_API_KEY,
        },
    },
}

# Only add LLM config if we have one
if llm_config:
    config["llm"] = llm_config

# Debug summary
print(f"[mem0] Embeddings: OpenAI text-embedding-3-small")

# Initialize mem0
MEM0 = None
if OPENAI_API_KEY:
    try:
        _clear_mem0_env_vars()
        MEM0 = Memory.from_config(config)
        print("[mem0] Memory initialized successfully")
    except Exception as e:
        print(f"[mem0] WARNING: Failed to initialize Memory: {e}")
        print("[mem0] App will run without memory features")
    finally:
        _restore_env_vars()
else:
    print("[mem0] OPENAI_API_KEY not set - mem0 disabled (no embeddings)")
