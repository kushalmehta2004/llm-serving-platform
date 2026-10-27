"""
Phase 1 — single-request OpenAI-compatible completion endpoint.
Phase 2 — stream=true via SSE.
Phase 3 — requests go through the queue + static batching scheduler.
"""

import json
import os
import time
import uuid
import asyncio
from typing import AsyncGenerator, Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from server.model import load_model, get_tokenizer
from server.queue import RequestQueue, Request
from server.scheduler import ContinuousBatchScheduler, StaticBatchScheduler

load_dotenv()

app = FastAPI(title="LLM Serving Platform")

_request_queue: RequestQueue | None = None
_scheduler = None


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
    global _request_queue, _scheduler
    load_model()
    _request_queue = RequestQueue()
    max_batch = int(os.environ.get("MAX_BATCH", 8))
    scheduler_type = os.environ.get("SCHEDULER", "continuous")
    if scheduler_type == "static":
        _scheduler = StaticBatchScheduler(_request_queue, max_batch=max_batch)
    else:
        _scheduler = ContinuousBatchScheduler(_request_queue, max_batch=max_batch)
    _scheduler.start()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _messages_to_prompt(messages: list[ChatMessage]) -> str:
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


def _make_request(req: ChatCompletionRequest, prompt: str) -> Request:
    return Request(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        seed=req.seed,
        output_queue=asyncio.Queue(),
        loop=asyncio.get_event_loop(),
    )


async def _collect_tokens(output_queue: asyncio.Queue) -> list[int]:
    """Drain the output queue until the None sentinel, return all token ids."""
    tokens = []
    while True:
        tok = await output_queue.get()
        if tok is None:
            break
        tokens.append(tok)
    return tokens


async def _stream_from_queue(
    req: ChatCompletionRequest,
    output_queue: asyncio.Queue,
    completion_id: str,
    created: int,
) -> AsyncGenerator[str, None]:
    tokenizer = get_tokenizer()
    while True:
        tok = await output_queue.get()
        if tok is None:
            break
        token_text = tokenizer.decode([tok], skip_special_tokens=True)
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

    request = _make_request(req, prompt)
    _request_queue.enqueue(request)

    if req.stream:
        return StreamingResponse(
            _stream_from_queue(req, request.output_queue, completion_id, created),
            media_type="text/event-stream",
        )

    # Non-streaming: wait for all tokens then return
    token_ids = await _collect_tokens(request.output_queue)
    tokenizer = get_tokenizer()

    # prompt_tokens: re-tokenize to get count
    prompt_token_count = len(tokenizer(prompt)["input_ids"])

    text = tokenizer.decode(token_ids, skip_special_tokens=True)
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
            prompt_tokens=prompt_token_count,
            completion_tokens=len(token_ids),
            total_tokens=prompt_token_count + len(token_ids),
        ),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
