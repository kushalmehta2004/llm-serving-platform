"""
Phase 3 — static batching scheduler.

The loop:
  1. Block until at least one request arrives.
  2. Drain up to MAX_BATCH requests without further blocking.
  3. Run the whole batch to completion via model.generate() (left-padded).
  4. Stream each sequence's tokens back to its requester.
  5. Repeat.

"Naive" = the batch is locked together until the slowest request finishes.
A short request waits for the longest one before its slot is freed.
Phase 4 fixes this with a continuous step-by-step eviction loop.
"""

import threading

from server.queue import RequestQueue, Request
from server.model import batch_generate


class StaticBatchScheduler:
    def __init__(self, request_queue: RequestQueue, max_batch: int = 8):
        self.request_queue = request_queue
        self.max_batch = max_batch
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    # ── main scheduler loop ───────────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            requests = self.request_queue.dequeue_batch(self.max_batch)
            try:
                self._run_batch(requests)
            except Exception as exc:
                for req in requests:
                    _send(req, None)
                print(f"[scheduler] batch error: {exc}")

    # ── batch execution ───────────────────────────────────────────────────────

    def _run_batch(self, requests: list[Request]) -> None:
        # Run all prompts together; model.generate() left-pads internally.
        # Sequences with different max_tokens: use the maximum so every
        # sequence gets its requested number of tokens. Shorter-limit sequences
        # stop when they hit their personal limit or EOS.
        from server.model import get_tokenizer
        tokenizer = get_tokenizer()

        prompts    = [r.prompt     for r in requests]
        max_new    = max(r.max_tokens for r in requests)
        # Use temperature 0 if all are greedy, else just use the first request's
        # temperature (mixed-temp batching is a Phase 4 concern)
        temperature = requests[0].temperature

        all_token_ids = batch_generate(prompts, max_new, temperature)

        for req, token_ids in zip(requests, all_token_ids):
            # Respect per-request max_tokens (EOS already stripped by batch_generate)
            for tok in token_ids[: req.max_tokens]:
                _send(req, tok)
            _send(req, None)   # done sentinel


# ── helper ────────────────────────────────────────────────────────────────────

def _send(req: Request, value) -> None:
    """Thread-safe: post a value into the asyncio queue on the right loop."""
    req.loop.call_soon_threadsafe(req.output_queue.put_nowait, value)
