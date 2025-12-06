from __future__ import annotations

import os
from mem0 import Memory

# Option 1: default config (OpenAI + local Qdrant)
# Requires OPENAI_API_KEY in env
MEM0 = Memory()

# Option 2 (later): if you want to fully customize (Ollama/HF/etc),
# you can replace the above with:
#
# from mem0 import Memory
#
# config = {
#     "vector_store": {
#         "provider": "qdrant",
#         "config": {
#             "collection_name": "clara_memories",
#             "host": "localhost",
#             "port": 6333,
#             "embedding_model_dims": 768,
#         },
#     },
#     "llm": {
#         "provider": "ollama",
#         "config": {
#             "model": "llama3.1:latest",
#             "temperature": 0,
#         },
#     },
#     "embedder": {
#         "provider": "ollama",
#         "config": {
#             "model": "nomic-embed-text:latest",
#         },
#     },
# }
#
# MEM0 = Memory.from_config(config)
