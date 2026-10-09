"""Gemini client. LLM use is narrow: exception explanation, commentary, text-to-SQL, policy RAG.

Every helper returns None when no API key is set so the deterministic pipeline
still runs end to end (and tests stay offline).
"""
from __future__ import annotations

import os
import time
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
T = TypeVar("T", bound=BaseModel)
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
last_error: str | None = None  # last swallowed API error, so callers can report it instead of failing silently


def _record(e: Exception) -> None:
    global last_error
    last_error = f"{type(e).__name__}: {str(e)[:300]}"


def available() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


def _model():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL), temperature=0)


def structured(prompt: str, schema: type[T], retries: int = 3) -> T | None:
    if not available():
        return None
    llm = _model().with_structured_output(schema)
    for i in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:  # rate limit / transient; free tier is spiky
            _record(e)
            time.sleep(2 ** i)
    return None


def text(prompt: str, retries: int = 3) -> str | None:
    if not available():
        return None
    llm = _model()
    for i in range(retries):
        try:
            return llm.invoke(prompt).text
        except Exception as e:
            _record(e)
            time.sleep(2 ** i)
    return None


def embed(texts: list[str], retries: int = 3) -> list[list[float]] | None:
    if not available():
        return None
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    emb = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    for i in range(retries):
        try:
            return emb.embed_documents(texts)
        except Exception as e:
            _record(e)
            time.sleep(2 ** i)
    return None
