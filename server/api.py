"""
Phase 1 — single-request OpenAI-compatible completion endpoint (non-streaming).
Phase 2 will add stream=true handling.
"""

import time
import uuid
import os
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from server.model import load_model, generate

load_dotenv()

app = FastAPI(title="LLM Serving Platform")


# ── Request / response schemas (OpenAI-compatible) ──────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default_factory=lambda: os.environ.get("MODEL_ID", "unknown"))
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 1.0
    stream: bool = False
    seed: int | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_model()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Convert chat messages to a plain text prompt using the tokenizer's chat template when available."""
    from server.model import get_tokenizer
    tokenizer = get_tokenizer()
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [m.model_dump() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback: simple concatenation
    parts = []
    for m in messages:
        parts.append(f"{m.role}: {m.content}")
    parts.append("assistant:")
    return "\n".join(parts)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    prompt = _messages_to_prompt(req.messages)
    text, _token_ids, prompt_tokens, completion_tokens = generate(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        seed=req.seed,
    )
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
