"""
Discord bot for Clara - Multi-user AI assistant with memory.

Inspired by llmcord's clean design, but integrates directly with Clara's
MemoryManager for full mem0 memory support.

Usage:
    poetry run python discord_bot.py

Environment variables:
    DISCORD_BOT_TOKEN - Discord bot token (required)
    DISCORD_CLIENT_ID - Discord client ID (for invite link)
    DISCORD_MAX_MESSAGES - Max messages in conversation chain (default: 25)
    DISCORD_MAX_CHARS - Max chars per message content (default: 100000)
    DISCORD_ALLOWED_CHANNELS - Comma-separated channel IDs (optional, empty = all)
    DISCORD_ALLOWED_ROLES - Comma-separated role IDs (optional, empty = all)
"""

from __future__ import annotations

import os

# Load .env BEFORE other imports that read env vars at module level
from dotenv import load_dotenv

load_dotenv()

import asyncio
import io
import json
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import discord
import uvicorn
from discord import Message as DiscordMessage
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from db import SessionLocal
from db.models import ChannelSummary, Project, Session
from sandbox.docker import get_sandbox_manager
from storage.local_files import get_file_manager
from email_monitor import (
    handle_email_tool,
    email_check_loop,
)
from config.logging import init_logging, get_logger, set_db_session_factory

# Import modular tools system for GitHub, ADO, etc.
from tools import init_tools, get_registry, ToolContext

# Import from clara_core for unified platform
from clara_core import (
    init_platform,
    MemoryManager,
    make_llm,
    make_llm_with_tools,
    ModelTier,
    get_model_for_tier,
)

# Initialize logging system
init_logging()
logger = get_logger("discord")
tools_logger = get_logger("tools")

# Configuration
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CLIENT_ID = os.getenv("DISCORD_CLIENT_ID", "")
MAX_MESSAGES = int(os.getenv("DISCORD_MAX_MESSAGES", "25"))
MAX_CHARS = int(os.getenv("DISCORD_MAX_CHARS", "100000"))
MAX_FILE_SIZE = int(os.getenv("DISCORD_MAX_FILE_SIZE", "100000"))  # 100KB default
SUMMARY_AGE_MINUTES = int(os.getenv("DISCORD_SUMMARY_AGE_MINUTES", "30"))
CHANNEL_HISTORY_LIMIT = int(os.getenv("DISCORD_CHANNEL_HISTORY_LIMIT", "50"))
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "America/New_York")