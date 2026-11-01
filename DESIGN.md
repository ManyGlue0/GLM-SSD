# GLM-SSD — design

An inference engine, written from scratch, for **GLM-5.3-Flash** (320B total / 18B active MoE).
It streams routed experts from NVMe and caches the hot ones in RAM and VRAM.
Target model file: `unsloth/GLM-5.3-Flash-GGUF`, quant `UD-IQ2_XXS` (101.84 GB, 4 shards).

## Target machine

| | |
|---|---|
| GPU | RTX 3090 24 GB (sm_86), PCIe 4.0 x16 (~25 GB/s H2D with pinned memory) |
| CPU | Ryzen 9 5950X, 16C/32T |
| RAM | 32 GB DDR4 (currently 2133 MT/s — enabling XMP/DOCP is free bandwidth) |
| SSD | Samsung 990 PRO 2 TB (E:), ~7 GB/s sequential, excellent at QD≥8 random 1–4 MB reads |
| Toolchain | Rust 1.99 (MSVC), CUDA 12.9, VS 2022 |

## The model (from the GGUF headers)

`glm5-next`: 45 trunk layers + 1 MTP (NextN) layer, hidden 4096, vocab 154 880.

- **Residual stream = mHC** (manifold-constrained hyper-connections): 4 parallel copies of the
  4096-d stream. Before each sub-block: RMSNorm(flatten 4×4096) → `hc_fn` (16384→24) gives
  `pre`(4, sigmoid+eps), `post`(4, 2·sigmoid), `comb`(4×4, Sinkhorn, 20 iters).
  Input to the block = Σ pre·streams; after it: stream_d = post_d·out + Σ_s comb[d,s]·stream_s.
  The final head uses the mean of the 4 streams.
- **Attention, 34 layers = KDA** (Kimi Delta Attention, linear): 64 heads × 128.
  Q/K/V projections → causal short conv (k=4) → SiLU, L2-normed q/k, per-channel decay gate
  (low-rank f_a/f_b, lower bound −5), β = sigmoid, gated delta-rule recurrence with a
  **fixed 64×128×128 state per layer** (~4 MB fp32). Output: RMSNorm·sigmoid(g) → o_proj.
- **Attention, 11 layers (3,7,…,43) = nope-MLA + DSA sparse indexer**: q_lora 1536,
  kv_lora 512, 64 heads × 256, **no RoPE**. Only the 512-d latent is cached (absorbed wk_b/wv_b).
  The lightning indexer (32 heads × 128) scores **pools of 4 tokens** (gated soft pooling)
  and picks the top 2048/4 = 512 pools + the incomplete tail → attention over ≤2051 tokens.
- **FFN**: layers 0–2 dense (12288); layers 3–44 MoE: 288 experts, top-8, sigmoid routing
  with selection bias (`noaux_tc`), normalized weights × 2.5, + 1 shared expert (2048),
  SwiGLU clamped at 10.
- **MTP layer 45**: eh_proj(concat(norm(emb), norm(h))) → DSA attn → MoE → shared head.

### Tensor bytes (UD-IQ2_XXS)

| Part | Size | Placement |
|---|---|---|
| Routed experts (42 layers × 288 + MTP) | ~95 GB | **streamed**: SSD → RAM cache → VRAM cache |
| One expert (gate+up+down) | ~7.5 MB typical (IQ2_XXS/IQ2_XXS/IQ3_XXS) | unit of caching & I/O |
| Everything else (attn, routers, shared experts, dense FFN, head) | ~6.5 GB | **resident in VRAM** |
| Token embedding (Q4_K, 357 MB) | | RAM (row lookup on CPU) |

Per generated token: 42 layers × 8 experts × 7.5 MB ≈ **2.5 GB of expert weights touched**.
Caches can hold ~30–40 GB of the 95 GB, so the speed depends on the cache hit rate and on how
much of the remaining SSD traffic can be hidden behind compute.

## Architecture

```
            ┌───────────────────────── GPU (RTX 3090) ──────────────────────────┐
 tokens ──► │ resident weights (~6.5 GB) │ expert VRAM cache (slab of slots)    │
            │ KV/state caches             │ compute: custom CUDA kernels (IQ*)  │
            └───────────▲─────────────────┴──────────────▲───────────────────────┘
                        │ H2D (pinned, async, own stream)│
            ┌───────────┴────────────────────────────────┴───────────────────────┐
            │ RAM expert cache (pinned slots)  ◄── unbuffered overlapped reads ── │ 990 PRO
            └─────────────────────────────────────────────────────────────────────┘
```

- **Host language**: Rust (`engine`), kernels in CUDA C (`kernels/*.cu`) built by nvcc through
  `build.rs`, called via a thin `extern "C"` FFI.
- **Weights format**: read the GGUF shards directly (no repack). Each expert = 3 extents
  (gate, up, down slices of the 3-D tensors). Reads use `FILE_FLAG_NO_BUFFERING` with
  4 KiB-aligned windows (slack is discarded), overlapped I/O, queue depth ~32.
- **Expert cache** (two tiers): fixed-size slots, keyed by (layer, expert). Admission/eviction:
  frequency with decay (LFU-ish) + recency; seeded at start-up by the **learned hot-expert profile**.
- **Hiding I/O**: after computing a layer's router, its misses are issued immediately; hits compute
  while misses stream in. **Speculative prefetch**: run layer L+1's router on layer L's hidden
  state to start loading likely experts one layer early.
- **Prefill**: process big chunks of tokens per layer, so every loaded expert serves many
  tokens (I/O is amortized → hundreds of tok/s).
- **Budgets**: `--vram-budget` / `--ram-budget` flags (required by design, no presets).

## Roadmap

1. **M0** workspace, GGUF reader, `inspect` placement planner, CUDA build pipeline. ✅
2. **M1** dequant + matvec kernels for every quant type in the file
   (Q4_K Q5_K Q6_K Q8_0 IQ2_XXS IQ2_S IQ3_XXS IQ4_XS Q2_K Q3_K F32/F16). ✅
   `glm-ssd qtest` checks them bit-exact against the CPU port (`crates/quants`) on random and
   real blocks; `glm-ssd qbench` measures GEMV bandwidth (IQ2_XXS ~200-270 GB/s, IQ3_XXS ~400,
   Q8_0/F16 ~800 of 936 GB/s at batch 1). Batch 4-8 is input-load bound -> int8 x + dp4a in M5.
3. **M2** correct forward pass (slow path) — mHC, KDA, nope-MLA+indexer, MoE, head.
   Validate logits against llama.cpp master (which supports `glm5-next`) on fixed prompts.
4. **M3** the streaming engine: async SSD reader, two-tier expert cache, prefetch, budgets.
5. **M4** tokenizer (GLM BPE from GGUF) + chat template + sampler + **CLI chat**.
6. **M5** chunked prefill, CUDA graphs for decode.
7. **M6** OpenAI-compatible HTTP server (streaming).
8. **M7** prompt/prefix caching (MLA latents + KDA/conv state snapshots at checkpoints).
9. **M8** MTP speculative decoding.
10. **M9** learned hot-expert profile (persisted usage stats → preload).
11. **M10** long-context: KDA chunked scan for prefill, efficient indexer/top-k, 128K+ contexts.

## References (read-only, in `ref/`, not part of the build)

- llama.cpp master `src/models/glm5-next.cpp` (+ delta-net, kpool cache) — MIT.
- transformers `models/glm5_next/modeling_glm5_next.py` — Apache-2.0.
- IQ quant grids/tables are ported from ggml (MIT) with attribution.
