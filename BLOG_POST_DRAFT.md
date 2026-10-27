# I Built an LLM Server from Scratch, Then Compared It to vLLM

_A walkthrough of continuous batching and paged KV caching — what they are, why they matter, and what it costs to close the gap with a production system._

---

## The Gap I Wanted to Close

Most ML engineers can call `vllm.LLM()` or hit an OpenAI endpoint. Far fewer know what happens inside — why batching matters, what a KV cache actually is, or why naive serving collapses under load.

I built a from-scratch LLM inference server to close that gap for myself. No vLLM imports in the server code. Just PyTorch, Transformers (for the model + tokenizer), and original batching and cache logic. Then I benchmarked it against vLLM across concurrency levels to see exactly where the gap is and why.

---

## The Problem with Naive Serving

The simplest possible LLM server processes one request at a time: tokenize, run the model, decode token by token, return. It's correct, but wasteful — the GPU sits ~90% idle waiting on memory transfers and sequential token generation.

The second instinct is static batching: collect N requests, pad them all to the same length, run a single batched forward pass per decode step. This helps throughput, but creates a new problem: **a short request is held hostage by long ones**.

If request A generates 200 tokens and request B generates 10, B must wait for A to finish before its slot is freed — even though B is done after step 10. In my benchmark:

| Scheduler | Short-request latency | Long-request latency |
|-----------|----------------------|----------------------|
| Static batching | 12.2 s | 12.2 s |
| Continuous batching | **4.8 s** | 27.4 s |

The short request gets **2.55× lower latency** under continuous batching.

---

## Continuous Batching: The Key Idea

The insight is simple: **don't wait for the full batch to finish**. Instead, run one decode step across all active sequences. When any sequence emits an EOS token, evict it immediately and admit the next waiting request into its slot.

```
step 1: [A_tok1, B_tok1, C_tok1]   ← 3 active sequences
step 2: [A_tok2, B_tok2, C_tok2]
step 3: [A_tok3, B done  , C_tok3]  ← B finishes, slot freed
step 4: [A_tok4, D_tok1  , C_tok4]  ← D admitted immediately
```

The GPU never idles waiting for a long request to finish before admitting a short one. This is the core loop in `server/scheduler.py`:

```python
while True:
    # admit new requests into free slots
    new_reqs = self._admit(free_slots, block=len(active) == 0)
    for req in new_reqs:
        active.append(ActiveSeq(req, tokenize(req.prompt)))

    # one batched forward pass
    input_ids, attn_mask = build_left_padded_batch(active)
    logits = model(input_ids, attn_mask).logits[:, -1, :]

    # emit one token per sequence, evict finished
    for seq in active:
        tok = logits[seq.idx].argmax()
        emit(seq.request, tok)
        if done(seq, tok):
            evict(seq)
```

No vLLM. ~60 lines of Python.

---

## The KV Cache Problem

Each decode step re-computes the full attention over all previous tokens — or it would, without a KV cache. The KV cache stores the key and value tensors from prior steps so the model only attends over the new token at each step.

The naive approach: allocate a contiguous tensor of shape `(max_seq_len, num_layers, 2, num_kv_heads, head_dim)` per sequence. On admission. Before you know how long it will be.

This wastes VRAM two ways:
1. **Internal fragmentation:** a 10-token response holds a 2048-token buffer.
2. **External fragmentation:** freeing a long sequence leaves a gap too large for most new sequences to fill.

---

## Paged KV Cache: OS Paging for Attention Memory

The solution is borrowed directly from OS virtual memory. Instead of one contiguous buffer per sequence, divide VRAM into fixed-size **blocks** (e.g., 8 tokens each) from a shared pool. Each sequence gets a **block table** mapping logical block positions to physical blocks.

```
BlockPool:  [blk_0] [blk_1] [blk_2] ... [blk_63]   ← 64 × 8 = 512 token-slots
SequenceA:  logical 0 → blk_5,  logical 1 → blk_22
SequenceB:  logical 0 → blk_1
```

Benefits:
- Sequences only hold blocks they've **actually filled** — no pre-reservation
- Freed blocks go straight back to the pool — no fragmentation
- Higher effective concurrency: more sequences fit in the same VRAM budget

My implementation (`server/kv_cache.py`):

```python
class BlockPool:
    # data: (num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim)
    def alloc(self) -> int: ...
    def free(self, block_id: int) -> None: ...

class SequenceKV:
    def write_token(self, past_kv: DynamicCache, token_pos: int): ...
    def to_dynamic_cache(self) -> DynamicCache: ...  # gather pages → contiguous
    def free(self): ...  # return blocks to pool
```

The catch: without a custom CUDA kernel (PagedAttention), `to_dynamic_cache()` gathers the paged blocks back into a contiguous tensor before each forward pass. The memory management is correct; the kernel-level speedup (avoiding the gather) is what vLLM adds.

---

## Where vLLM Wins and Why

After benchmarking, here's the honest accounting of the remaining gap:

| Factor | From-scratch | vLLM |
|--------|-------------|------|
| Batching strategy | Continuous | Continuous |
| KV cache layout | Paged (gather before forward) | Paged (fused attention kernel) |
| Attention kernel | PyTorch stock | PagedAttention CUDA kernel |
| Prefill/decode split | No | Yes (chunked prefill) |
| Speculative decoding | No | Optional |
| Tensor parallelism | No | Yes |

The single biggest gap is the fused PagedAttention kernel. My `to_dynamic_cache()` gather is O(T) on every decode step; vLLM's kernel reads directly from the block table without materialising a contiguous tensor.

The rest of the gap (chunked prefill, speculative decoding, tensor parallelism) is engineering effort, not a fundamental barrier. The batching and caching *concepts* are identical.

---

## The Kubernetes War Story

Deploying to K8s revealed one sharp edge: the model takes 30–120 seconds to load. A naive readiness probe with `initialDelaySeconds: 10` causes `CrashLoopBackOff` — the kubelet fires the probe before the model is ready, gets no response, restarts the pod, repeat.

The fix is a `startupProbe`:

```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8000
  failureThreshold: 30   # 30 × 10s = 300s budget
  periodSeconds: 10
```

`startupProbe` blocks `livenessProbe` and `readinessProbe` from running until it first succeeds. The pod becomes Ready exactly when the model finishes loading — no blind wait, no restart loop.

---

## What I Learned

1. **Continuous batching is ~60 lines.** The concept sounds complex; the implementation is a loop with a left-padded batch and an eviction check.

2. **Paged KV is memory management, not attention magic.** The hard part is the fused CUDA kernel vLLM adds on top. The block-table bookkeeping is straightforward.

3. **Probes need budgets, not delays.** `initialDelaySeconds` is a guess. `startupProbe.failureThreshold × periodSeconds` is a contract.

4. **Building from scratch makes benchmarks meaningful.** I can explain every line of the throughput curve because I wrote every line of the server.

---

## Links

- [GitHub repo](https://github.com/kushalmehta2004/llm-serving-platform)
- Benchmark charts: `bench/charts/` (generated by `python bench/plot.py`)
- W&B runs: `llm-serving-platform` project
