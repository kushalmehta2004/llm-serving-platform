"""
Phase 1 — single-request OpenAI-compatible completion endpoint.
Phase 2 — stream=true via SSE (text/event-stream).
"""

import json
import time
import uuid
import os
from typing import AsyncGenerator, Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from server.model import load_model, generate, generate_stream

load_dotenv()

app = FastAPI(title="LLM Serving Platform")


# ── Request / response schemas (OpenAI-compatible) ───────────────────────────

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


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_model()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    from server.model import get_tokenizer
    tokenizer = get_tokenizer()
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [m.model_dump() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = [f"{m.role}: {m.content}" for m in messages]
    parts.append("assistant:")
    return "\n".join(parts)


async def _stream_response(
    req: ChatCompletionRequest,
    prompt: str,
    completion_id: str,
    created: int,
) -> AsyncGenerator[str, None]:
    """Yields OpenAI-style SSE chunks then a [DONE] sentinel."""
    from server.model import get_tokenizer
    tokenizer = get_tokenizer()

    for token_id, _prompt_len in generate_stream(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        seed=req.seed,
    ):
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": token_text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk with finish_reason
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    prompt = _messages_to_prompt(req.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _stream_response(req, prompt, completion_id, created),
            media_type="text/event-stream",
        )

    # Non-streaming path (Phase 1) — unchanged
    text, _token_ids, prompt_tokens, completion_tokens = generate(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        seed=req.seed,
    )
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
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
