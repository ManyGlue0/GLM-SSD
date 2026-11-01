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

## Measurements

**SSD (990 PRO, `glm-ssd iobench`, unbuffered random reads):**

| read size | 1 thread | best |
|---|---|---|
| 7.2 MiB (whole expert) | 5.49 GB/s, 1.4 ms | 5.72 GB/s (4 threads) |
| 2.1 MiB (one gate/up slice) | 4.61 GB/s, 0.5 ms | 4.87 GB/s (2 threads) |

- One expert costs ~1.3 ms from SSD; the drive saturates at queue depth 1-2 for multi-MB reads,
  so the reader needs few threads and should keep requests large.
- Per-expert-contiguous storage (one read instead of three) would gain ~15% -> optional one-time
  repack in M3 (costs ~95 GB extra disk while the GGUF is kept for reference runs).
- Uncached worst case: 336 experts x 1.3 ms = 0.44 s/token (~2.3 tok/s). Every cache hit removes
  1.3 ms; at a 60% hit rate the SSD floor is ~0.18 s/token (~5.5 tok/s) before overlap with compute.

**GEMV (`glm-ssd qbench`, batch 1):** IQ2_XXS ~200-270 GB/s, IQ3_XXS ~400 GB/s, Q8_0/F16 ~800 GB/s
-> all 336 routed experts of one token take ~8 ms of GPU time once they are in VRAM.

## Roadmap

1. **M0** workspace, GGUF reader, `inspect` placement planner, CUDA build pipeline. ✅
2. **M1** dequant + matvec kernels for every quant type in the file
   (Q4_K Q5_K Q6_K Q8_0 IQ2_XXS IQ2_S IQ3_XXS IQ4_XS Q2_K Q3_K F32/F16). ✅
   `glm-ssd qtest` checks them bit-exact against the CPU port (`crates/quants`) on random and
   real blocks; `glm-ssd qbench` measures GEMV bandwidth (IQ2_XXS ~200-270 GB/s, IQ3_XXS ~400,
   Q8_0/F16 ~800 of 936 GB/s at batch 1). Batch 4-8 is input-load bound.
3. **M2** correct forward pass (slow path) — mHC, KDA, nope-MLA+indexer, MoE, head. ✅
   Validated 2026-11-01: `glm-ssd run --trace` tracks `llama-eval-callback` within 1-3% through
   all 45 layers (that gap is llama.cpp's Q8_1 activation quantization: `glm-ssd ref-l0`, an
   independent f64 CPU reference, agrees with our GPU path to 6 digits on layer 0 while llama.cpp
   is off by up to 20% where KDA's k.q is ill-conditioned). Chat prompt "What is the capital of
   France?" -> "The capital of France is **Paris**." + EOS. ~1.1 tok/s with no expert cache.
   Tokenizer (byte-level BPE, chatglm4 regex): `glm-ssd tokenize`.
4. **M3** streaming engine ✅ (`experts.rs`): VRAM + pinned-RAM expert slots per size class,
   decayed-frequency eviction (half-life ~350 tokens), RAM evicts experts already in VRAM first
   (mostly exclusive tiers without D2H copies), direct I/O reader pool (8 threads, each expert's
   3 extents read in parallel, demand > prefetch priority), copy stream + events so uploads
   overlap compute, decode prefetch of the next layer's top-2 predicted experts (96% accurate).
   `--vram-budget` is the engine's total VRAM; what weights/caches/scratch leave is expert cache.
5. **M4** CLI chat ✅: the GGUF's Jinja template rendered by minijinja (+pycompat), streaming
   detokenizer that splits reasoning / content / tool calls, sampler (temp, top-k/p, min-p).
6. **M5** batched prefill ✅: the forward pass runs T tokens per call; T > 8 uses a tensor-core
   GEMM that dequantizes weight tiles on the fly (WMMA f16, f32 accumulation; checked against
   the CPU reference for every quant type in `qtest`). MoE groups tokens per expert, so each
   expert is read once per chunk. CUDA graphs were dropped: decode syncs with the host at every
   MoE layer to route (the host must issue the SSD reads), and launch overhead is ~3% of a token.
7. **M6** OpenAI-compatible server ✅ (`serve`): /v1/chat/completions (SSE streaming with
   `reasoning_content` deltas, tools <-> GLM `<tool_call>` XML), /v1/completions, /v1/models.
8. **M7** prefix caching ✅ (`session.rs`): attention caches are position-indexed, so a shared
   prefix is reused as-is; KDA/conv states + indexer rings + the MTP pending row are snapshotted
   (~150 MB, host RAM) at the end of every prompt and every 8K prefill tokens; a request restores
   the newest snapshot at or before its divergence point.
9. **M8** MTP speculative decoding ✅ (`--mtp`): the NextN layer drafts 1 token, the trunk verifies
   [cur, draft] in one T=2 pass; on rejection the KDA/conv states saved after the first token are
   restored. The MTP layer's attention cache is fed during prefill too. Exact for any sampler
   (accept iff the sampled token equals the draft).
10. **M9** hot-expert profile ✅: lifetime use counts in `hot-experts.txt` seed the eviction
    scores; `--preload` fills both tiers from it at start (~26 GB in 5.4 s).
11. **M10** long context ✅: f16 MLA latent cache, indexer raw rows kept only in a 4-row ring
    (pooled keys per 4 tokens are what persists), GPU indexer scoring + exact radix top-k per
    query, flash-style MLA attention over the selected positions. The KDA prefill scan keeps each
    head's 128x128 state in registers and walks tokens sequentially (~ms per layer per chunk), so
    a chunked/WY scan was not needed. 131K context allocates fine on a 24 GB card.

### Results (RTX 3090, 990 PRO, 22G VRAM / 12G RAM budgets, cold caches, no profile)

| | |
|---|---|
| decode, steady state | 5.6 tok/s (5.9 with `--mtp`, 81% accepted on prose, 98% on lists) |
| prefill | 58 tok/s (chunk 1024, 3.5K prompt), 122 tok/s (chunk 2048, 17K prompt) |
| long context | needles retrieved at 3.5K and 17K tokens (sparse DSA path), ctx 131072 |
| expert hit rate (decode) | ~60% VRAM + ~14% RAM, rest from SSD at ~6 GB/s |

### Speed vs quality options (decode, `glm-ssd ppl` on 300 tokens of prose, exact = 3.9917 ppl)

Exact routing is deterministic, so the perplexity differences below are real, not noise.

| option | decode | ppl | SSD share of expert uses |
|---|---|---|---|
| exact | 4.54 tok/s | 3.9917 | 23.1% |
| `--kv-cache q8` | 4.61 | 3.9707 | 23.0% |
| `--cache-aware 0.05` | 6.42 | 3.9729 | 12.3% |
| `--cache-aware 0.1` | 7.51 | 3.9944 | 8.1% |
| `--cache-aware 0.2` | 8.92 | 4.0446 | 4.5% |
| `--cache-aware 0.4` | 9.68 | 4.0764 | 2.5% |
| `--mtp --cache-aware 0.1` (bench) | 9.36 | | 14% |

Eviction half-life (GLMSSD_DECAY 0.99 / 0.998 / 0.9995) does not change the hit rate.
Context memory: 12.7 KB/token with f16 latents (13.8 with MTP) -> 1.7-1.8 GB at 128K; Q8 saves 47%.

Decode is SSD-bandwidth bound (~0.8 GB of expert misses per token); more cache budget is the
main lever. Prefill streams nearly every expert once per chunk, so bigger chunks are faster.

## References (read-only, in `ref/`, not part of the build)

- llama.cpp master `src/models/glm5-next.cpp` (+ delta-net, kpool cache) — MIT.
- transformers `models/glm5_next/modeling_glm5_next.py` — Apache-2.0.
- IQ quant grids/tables are ported from ggml (MIT) with attribution.
