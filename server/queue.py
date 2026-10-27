"""
Thread-safe request queue.
Decouples HTTP arrival from the GPU's decode cadence — the mechanism
that makes batching possible.
"""

import queue
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Request:
    prompt: str
    max_tokens: int
    temperature: float
    seed: Optional[int]
    # Scheduler posts token_ids here; None = generation finished.
    output_queue: asyncio.Queue
    # The asyncio event loop the API handler is running on.
    loop: asyncio.AbstractEventLoop
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class RequestQueue:
    def __init__(self):
        self._q: queue.Queue[Request] = queue.Queue()

    def enqueue(self, request: Request) -> None:
        self._q.put(request)

    def dequeue_batch(self, max_batch: int) -> list[Request]:
        """
        Block until at least one request is available, then drain up to
        max_batch without blocking further.
        """
        batch = [self._q.get()]          # blocking — waits for work
        while len(batch) < max_batch:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break
        return batch
