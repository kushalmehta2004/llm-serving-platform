"""
Phase 4/5 — continuous batching scheduler (with optional paged KV cache).

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

import time

import torch

from server.queue import RequestQueue, Request
from server.model import get_tokenizer, get_model, get_model_device, decode_step_paged
from server.kv_cache import BlockPool, SequenceKV, make_pool_from_model

try:
    from server.metrics import tokens_generated, queue_depth, active_batch_size, kv_utilization, request_latency, ttft
    _METRICS = True
except Exception:
    _METRICS = False


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

            if _METRICS:
                active_batch_size.set(len(active))
                queue_depth.set(self.request_queue._q.qsize())

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
    if _METRICS and value is not None:
        tokens_generated.inc()
    req.loop.call_soon_threadsafe(req.output_queue.put_nowait, value)


# ── PagedContinuousBatchScheduler (Phase 5) ───────────────────────────────────

@dataclass
class PagedActiveSeq:
    request:    Request
    seq_kv:     SequenceKV
    prompt_len: int
    generated:  int = 0


class PagedContinuousBatchScheduler:
    """
    Continuous batching + paged KV cache.

    Differences from ContinuousBatchScheduler (Phase 4):
    • Each sequence maintains a SequenceKV (paged blocks) rather than
      accumulating the full token list.
    • Prefill and decode go through decode_step_paged(), which reads/writes
      KV via the block table.
    • On eviction, blocks are immediately returned to the pool so waiting
      sequences can use them.
    """

    def __init__(
        self,
        request_queue: RequestQueue,
        pool: BlockPool | None = None,
        max_batch: int = 8,
    ):
        self.request_queue = request_queue
        self.max_batch     = max_batch
        self._pool         = pool   # set before start() — or lazily in _loop
        self._thread       = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    @property
    def pool(self) -> BlockPool:
        return self._pool

    # ── main loop ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        from server.model import get_tokenizer, get_model_device
        import torch
        tokenizer = get_tokenizer()
        device    = get_model_device()
        eos_id    = tokenizer.eos_token_id

        # Lazy pool init (model must be loaded first)
        if self._pool is None:
            self._pool = make_pool_from_model()

        active: list[PagedActiveSeq] = []

        while True:
            # ── admit new requests ────────────────────────────────────────────
            free  = self.max_batch - len(active)
            block = len(active) == 0
            new_reqs = self._admit(free, block=block)

            for req in new_reqs:
                seq = self._prefill(req, tokenizer, device, eos_id, active)
                if seq is not None:
                    active.append(seq)

            if not active:
                continue

            if _METRICS:
                active_batch_size.set(len(active))
                queue_depth.set(self.request_queue._q.qsize())
                kv_utilization.set(self._pool.utilization)

            # ── decode one step per active sequence ───────────────────────────
            still_active: list[PagedActiveSeq] = []
            for seq in active:
                # cache_position: the position of the token we're about to generate
                cur_len    = seq.prompt_len + seq.generated
                cache_pos  = torch.tensor([cur_len], dtype=torch.long, device=device)
                inp        = torch.tensor([[seq._last_tok]], dtype=torch.long, device=device)
                attn_mask  = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)

                logits, _ = decode_step_paged(inp, attn_mask, seq.seq_kv, cache_pos)
                tok = int(logits[0].argmax().item())
                seq.generated    += 1
                seq._last_tok     = tok
                _send(seq.request, tok)

                done = (tok == eos_id or seq.generated >= seq.request.max_tokens)
                if done:
                    _send(seq.request, None)
                    seq.seq_kv.free()               # ← return blocks to pool
                else:
                    still_active.append(seq)

            active = still_active

    def _prefill(self, req: Request, tokenizer, device, eos_id: int, active: list) -> "PagedActiveSeq | None":
        """
        Tokenize, run prefill forward, store all prompt KV in paged blocks,
        emit first generated token.  Returns None if the sequence is done
        after the first token (i.e. EOS at step 0).
        """
        import torch
        from server.model import decode_step

        enc  = tokenizer(req.prompt, return_tensors="pt").to(device)
        ids  = enc["input_ids"]
        mask = enc["attention_mask"]
        L    = ids.shape[1]

        seq_kv    = SequenceKV(self._pool)
        cache_pos = torch.arange(L, dtype=torch.long, device=device)

        # Prefill: fill KV for all prompt tokens
        from transformers import DynamicCache
        past_kv   = DynamicCache()
        logits, past_kv = decode_step(ids, mask, past_kv, cache_pos)

        # Write ALL prompt tokens' KV to paged storage
        for tok_pos in range(L):
            seq_kv.write_token(past_kv, tok_pos)

        # First generated token
        tok = int(logits[0].argmax().item())
        _send(req, tok)

        if tok == eos_id or 1 >= req.max_tokens:
            _send(req, None)
            seq_kv.free()
            return None

        seq_obj           = PagedActiveSeq(request=req, seq_kv=seq_kv, prompt_len=L)
        seq_obj._last_tok = tok
        seq_obj.generated = 1
        # Also write the first generated token's KV — it's NOT in past_kv yet
        # because the first token is generated by the prefill logits, not a decode step.
        # We do a mini decode step to fill it:
        inp_first  = torch.tensor([[tok]], dtype=torch.long, device=device)
        mask_first = torch.ones((1, L + 1), dtype=torch.long, device=device)
        cp_first   = torch.tensor([L], dtype=torch.long, device=device)
        decode_step_paged(inp_first, mask_first, seq_kv, cp_first)

        return seq_obj

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
