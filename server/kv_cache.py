"""
Phase 5 — Paged KV Cache.

Analogy to OS paging: instead of each sequence owning a contiguous KV
tensor sized to its maximum length, memory is divided into fixed-size
BLOCKS shared from a global pool.  A per-sequence BLOCK TABLE maps
logical block positions to physical blocks in the pool.

Layout of the physical pool tensor:
  pool.data : (num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim)
                            ^--- 0=key, 1=value

Benefits over contiguous allocation
──────────────────────────────────
• No pre-reservation: a sequence only holds blocks it has FILLED so far,
  not blocks for future tokens it might generate.
• No fragmentation: freed blocks go straight back to the pool.
• With the same total token-slot budget, more concurrent short sequences
  can run simultaneously (shorter sequences return blocks early, letting
  new sequences immediately use them).

Limitations of this implementation
───────────────────────────────────
• No custom CUDA kernel: we gather paged blocks back into a contiguous
  tensor before each model forward pass.  The memory management is
  correct; the kernel-level speedup requires PagedAttention fused ops.
• Single-device only (device set at pool-init time).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import DynamicCache


# ── Physical block pool ───────────────────────────────────────────────────────

class BlockPool:
    """
    A flat pool of fixed-size KV blocks shared across ALL sequences.

    Parameters
    ----------
    num_blocks  : total physical blocks in the pool
    block_size  : tokens per block (e.g. 8)
    num_layers  : transformer layers
    num_kv_heads: KV attention heads (may differ from Q heads via GQA)
    head_dim    : per-head dimension
    device      : torch.device for all tensors
    """

    def __init__(
        self,
        num_blocks:   int,
        block_size:   int,
        num_layers:   int,
        num_kv_heads: int,
        head_dim:     int,
        device:       torch.device,
    ):
        self.num_blocks   = num_blocks
        self.block_size   = block_size
        self.num_layers   = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.device       = device

        # Physical storage: (num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim)
        self.data = torch.zeros(
            num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim,
            dtype=torch.float32, device=device,
        )

        self._free: list[int] = list(range(num_blocks))
        self._lock = threading.Lock()

    # ── allocation ────────────────────────────────────────────────────────────

    def alloc(self) -> int:
        with self._lock:
            if not self._free:
                raise RuntimeError(
                    f"KV block pool exhausted ({self.num_blocks} blocks all in use). "
                    "Reduce max_batch or increase num_blocks."
                )
            return self._free.pop()

    def free(self, block_id: int) -> None:
        with self._lock:
            self._free.append(block_id)

    # ── metrics ───────────────────────────────────────────────────────────────

    @property
    def num_free(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def utilization(self) -> float:
        """Fraction of blocks currently in use (0.0 – 1.0)."""
        with self._lock:
            used = self.num_blocks - len(self._free)
        return used / self.num_blocks

    def total_token_slots(self) -> int:
        return self.num_blocks * self.block_size


# ── Per-sequence paged KV state ───────────────────────────────────────────────

class SequenceKV:
    """
    Manages paged KV storage for one sequence.

    Blocks are allocated lazily — when the current block fills up, a new
    one is claimed from the pool.  All blocks are freed when the sequence
    completes.
    """

    def __init__(self, pool: BlockPool):
        self.pool        = pool
        self.block_ids:  list[int] = []   # logical block index → physical block id
        self.num_tokens: int = 0

    # ── writing ───────────────────────────────────────────────────────────────

    def _ensure_slot(self) -> tuple[int, int]:
        """Return (physical_block_id, slot_within_block) for the next write."""
        slot = self.num_tokens % self.pool.block_size
        if slot == 0:                        # current block full — get a new one
            self.block_ids.append(self.pool.alloc())
        return self.block_ids[-1], slot

    def write_token(self, past_kv: DynamicCache, token_pos: int) -> None:
        """
        Copy one token's K/V from past_kv (at position token_pos) for all
        layers into our paged storage, then advance num_tokens.

        Transformers 5.x: K/V stored in past_kv.layers[i].keys / .values
        Shape: (batch=1, num_kv_heads, seq_len, head_dim)

        Call once per token (not once per layer).
        """
        block_id, slot = self._ensure_slot()
        for layer_idx in range(self.pool.num_layers):
            k = past_kv.layers[layer_idx].keys[0, :, token_pos, :]    # (kv_heads, head_dim)
            v = past_kv.layers[layer_idx].values[0, :, token_pos, :]
            self.pool.data[block_id, 0, layer_idx, :, slot, :] = k.detach()
            self.pool.data[block_id, 1, layer_idx, :, slot, :] = v.detach()
        self.num_tokens += 1

    # ── reading ───────────────────────────────────────────────────────────────

    def to_dynamic_cache(self) -> DynamicCache:
        """
        Gather paged blocks back into a HuggingFace DynamicCache so the
        stock model forward pass can use it unchanged.

        Transformers 5.x: use cache.update(k, v, layer_idx) to populate.
        Returns a fresh DynamicCache with all num_tokens positions filled.
        """
        cache = DynamicCache()
        if self.num_tokens == 0:
            return cache

        cache_pos = torch.arange(self.num_tokens,
                                 dtype=torch.long,
                                 device=self.pool.device)

        for layer_idx in range(self.pool.num_layers):
            keys_chunks, values_chunks = [], []
            remaining = self.num_tokens

            for block_id in self.block_ids:
                valid = min(remaining, self.pool.block_size)
                keys_chunks.append(  self.pool.data[block_id, 0, layer_idx, :, :valid, :])
                values_chunks.append(self.pool.data[block_id, 1, layer_idx, :, :valid, :])
                remaining -= valid
                if remaining <= 0:
                    break

            # Concatenate along the seq-len dimension:
            # each chunk: (kv_heads, valid, head_dim) → cat → (kv_heads, T, head_dim)
            k = torch.cat(keys_chunks,   dim=1).unsqueeze(0)   # (1, kv_heads, T, head_dim)
            v = torch.cat(values_chunks, dim=1).unsqueeze(0)
            cache.update(k, v, layer_idx, cache_kwargs={"cache_position": cache_pos})

        return cache

    # ── cleanup ───────────────────────────────────────────────────────────────

    def free(self) -> None:
        """Return all held blocks to the pool.  Called on sequence eviction."""
        for block_id in self.block_ids:
            self.pool.free(block_id)
        self.block_ids.clear()
        self.num_tokens = 0


# ── Convenience: build a pool from a loaded model ────────────────────────────

def make_pool_from_model(num_blocks: int = 64, block_size: int = 8) -> BlockPool:
    """
    Build a BlockPool sized for the currently loaded model.

    Default: 64 blocks × 8 tokens = 512 total token-slots.
    For Qwen2.5-0.5B that's 64 × 8 × 2 × 24 × 2 × 64 × 4 bytes ≈ 25 MB.
    """
    from server.model import get_model, get_model_device
    model  = get_model()
    device = get_model_device()
    cfg    = model.config
    return BlockPool(
        num_blocks   = num_blocks,
        block_size   = block_size,
        num_layers   = cfg.num_hidden_layers,
        num_kv_heads = cfg.num_key_value_heads,
        head_dim     = cfg.hidden_size // cfg.num_attention_heads,
        device       = device,
    )
