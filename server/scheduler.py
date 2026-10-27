"""
Phase 4 — continuous batching scheduler.

Key difference from Phase 3 (static batching):
  Static batching: dequeue a batch, wait for ALL to finish, then dequeue again.
  Continuous batching: run a decode step every iteration; when any sequence
  finishes, admit a new request immediately so the batch stays full.

Under mixed-length workloads (some short, some long), this keeps compute
utilization high: short requests release their slots mid-batch.

Decode strategy: left-pad all active sequences to the same length, run ONE
batched model forward call per step.  This is equivalent to what Phase 3 does
per step but with continuous admission instead of batch-level blocking.

The separate KV-cache decode path (decode_step / DynamicCache) is validated in
tests/test_correctness.py::test_kv_cache_decode_matches_baseline.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

import torch

from server.queue import RequestQueue, Request
from server.model import get_tokenizer, get_model, get_model_device


# ── per-sequence bookkeeping ──────────────────────────────────────────────────

@dataclass
class ActiveSeq:
    request: Request
    token_ids: list[int]   # prompt tokens + generated tokens so far
    generated: int = 0


# ── scheduler ─────────────────────────────────────────────────────────────────

class ContinuousBatchScheduler:
    def __init__(self, request_queue: RequestQueue, max_batch: int = 8):
        self.request_queue = request_queue
        self.max_batch = max_batch
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    # ── main loop ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        tokenizer = get_tokenizer()
        model     = get_model()
        device    = get_model_device()
        pad_id    = tokenizer.pad_token_id or tokenizer.eos_token_id
        eos_id    = tokenizer.eos_token_id

        active: list[ActiveSeq] = []

        while True:
            # ── admit new requests into free slots ────────────────────────────
            free  = self.max_batch - len(active)
            block = len(active) == 0   # block when nothing is running
            new_reqs = self._admit(free, block=block)

            for req in new_reqs:
                enc = tokenizer(req.prompt, return_tensors="pt")
                token_ids = enc["input_ids"][0].tolist()
                active.append(ActiveSeq(request=req, token_ids=token_ids))

            if not active:
                continue

            # ── build left-padded batch ───────────────────────────────────────
            max_len = max(len(s.token_ids) for s in active)
            ids_list  = []
            mask_list = []
            for seq in active:
                ids = seq.token_ids
                pad = max_len - len(ids)
                ids_list.append([pad_id] * pad + ids)
                mask_list.append([0] * pad + [1] * len(ids))

            input_ids   = torch.tensor(ids_list,  dtype=torch.long, device=device)
            attn_mask   = torch.tensor(mask_list, dtype=torch.long, device=device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = out.logits[:, -1, :]  # (B, vocab)

            # ── emit token per sequence, evict finished ───────────────────────
            still_active: list[ActiveSeq] = []
            for i, seq in enumerate(active):
                tok = int(logits[i].argmax().item())
                seq.generated += 1
                seq.token_ids.append(tok)
                _send(seq.request, tok)

                done = (tok == eos_id or seq.generated >= seq.request.max_tokens)
                if done:
                    _send(seq.request, None)
                else:
                    still_active.append(seq)

            active = still_active

    # ── queue helpers ──────────────────────────────────────────────────────────

    def _admit(self, n: int, block: bool) -> list[Request]:
        if n <= 0:
            return []
        reqs: list[Request] = []
        if block:
            reqs.append(self.request_queue._q.get())
            n -= 1
        while n > 0:
            try:
                reqs.append(self.request_queue._q.get_nowait())
                n -= 1
            except Exception:
                break
        return reqs


# ── StaticBatchScheduler (Phase 3) ────────────────────────────────────────────

class StaticBatchScheduler:
    def __init__(self, request_queue: RequestQueue, max_batch: int = 8):
        self.request_queue = request_queue
        self.max_batch = max_batch
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        from server.model import batch_generate
        while True:
            batch = self.request_queue.dequeue_batch(self.max_batch)
            if not batch:
                continue
            prompts = [r.prompt for r in batch]
            max_tok = max(r.max_tokens for r in batch)
            results = batch_generate(prompts, max_tok, temperature=0.0)
            for req, tokens in zip(batch, results):
                for tok in tokens:
                    _send(req, tok)
                _send(req, None)


# ── helpers ───────────────────────────────────────────────────────────────────

def _send(req: Request, value) -> None:
    req.loop.call_soon_threadsafe(req.output_queue.put_nowait, value)
