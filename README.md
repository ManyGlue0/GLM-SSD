# GLM-SSD

An inference engine, written from scratch in Rust + CUDA, for **GLM-5.3-Flash** (unsloth
`UD-IQ2_XXS` GGUF, 102 GB). It runs on one 24 GB GPU by streaming the routed MoE experts from an
NVMe SSD and caching the hot ones in VRAM and pinned RAM.

See [DESIGN.md](DESIGN.md) for the architecture and measurements, [NOTES.md](NOTES.md) for
engineering notes.

## Build

Requirements: Windows or Linux, Rust (stable), CUDA 12.x, an sm_86 GPU (set `GLMSSD_CUDA_ARCH`
for others), MSVC on Windows.

```
cargo build --release
```

Download the model (resumable, SHA-256 verified):

```
.\download.ps1            # = cargo run --release -p hfdl -- (defaults to UD-IQ2_XXS into models/)
```

## Use

Memory budgets are always explicit:

- `--vram-budget` is the engine's total VRAM (weights + caches + scratch). Whatever is left
  becomes expert cache.
- `--ram-budget` covers the pinned expert cache and prompt-cache snapshots.

```
# interactive chat
glm-ssd chat --vram-budget 22G --ram-budget 12G --mtp

# OpenAI-compatible server on 127.0.0.1:8080
glm-ssd serve --vram-budget 22G --ram-budget 12G --mtp --ctx 65536

# decode benchmark
glm-ssd bench --vram-budget 22G --ram-budget 12G -n 200
```

Useful options:

| flag | meaning |
|---|---|
| `--ctx N` | max context (default 32768; 131072 fits on 24 GB) |
| `--chunk N` | prefill tokens per batched pass (default 512; 2048 roughly doubles prefill speed, costs ~1 GB VRAM) |
| `--mtp` | MTP speculative decoding |
| `--preload` | fill the expert caches from the learned hot-expert profile at start |
| `--profile FILE` / `--no-profile` | where usage statistics are kept (default `hot-experts.txt`) |
| `--prefetch K` | prefetch K predicted experts of the next layer while decoding (default 2) |
| `--cache-aware M` | cache-aware routing: an uncached expert of rank 3-8 is replaced by a cached one whose routing score is within M (default 0 = exact). 0.1 = +65% decode speed at unchanged perplexity on our test; 0.2 = +95% at +1.3% perplexity |
| `--kv-cache q8` | store the context (MLA latents) as Q8: 47% smaller (~0.85 GB saved at 128K), no measurable quality change |
| `--snapshots N` | prompt-cache snapshots kept in RAM (~150 MB each) |
| `--temp --top-p --top-k --min-p --seed` | sampling |

Chat commands: `/effort low|high|max`, `/think` (toggle reasoning display), `/reset`, `/stats`, `/exit`.

The server supports streaming (`reasoning_content` and `content` deltas), tools (OpenAI format,
mapped to GLM's `<tool_call>` XML), `reasoning_effort`, `stop`, and `stream_options.include_usage`.
Requests are served one at a time; consecutive requests that share a prefix (multi-turn chats,
agent loops) reuse it.

Diagnostics: `glm-ssd ppl --file text.txt` (perplexity + speed in decode mode, to compare
options), `inspect`, `qtest`, `qbench`, `iobench`, `tokenize`, `template`, `run --trace`
(compare with `llama-eval-callback`), `ref-l0` (f64 CPU reference of layer 0).

## Performance (RTX 3090 + Samsung 990 PRO, 22G / 12G budgets)

| | |
|---|---|
| decode, exact routing | ~5.6 tok/s, ~5.9 tok/s with `--mtp` |
| decode, `--mtp --cache-aware 0.1` | ~9.4 tok/s |
| prefill | ~60 tok/s (chunk 1024), ~120 tok/s (chunk 2048) |
