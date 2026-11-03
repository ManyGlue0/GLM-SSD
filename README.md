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

Install the `glm-ssd` command (into `~/.cargo/bin`, which is on PATH). Rerun it after pulling
changes. The binary finds the model in this repo's `models/` folder from any directory
(override with `-m` or `GLMSSD_MODEL`):

```
cargo install --path crates/engine
```

Download the model (resumable, SHA-256 verified):

```
.\download.ps1            # = cargo run --release -p hfdl -- (defaults to UD-IQ2_XXS into models/)
```

Other GLM-5.3-Flash GGUFs (community fine-tunes / re-quants with the same architecture) download
the same way from any Hugging Face repo, by URL or id:

```
.\download.ps1 --url https://huggingface.co/owner/name                       # picks the GGUF model
.\download.ps1 --url https://huggingface.co/owner/name/tree/main/UD-IQ2_XXS  # a folder
.\download.ps1 --url https://huggingface.co/owner/name/blob/main/X-00001-of-00004.gguf  # all shards of X
.\download.ps1 --repo owner/name --include "*IQ2_XXS*" --revision main
```

Without `--include`, a repo with one GGUF model (or one IQ2_XXS among several) is picked
automatically; otherwise the available models are listed with their sizes. Files go to
`models/<owner>--<name>/`, so they never overwrite the official ones. Then point `-m` at the
folder, or just at its name inside `models/`:

```
glm-ssd chat -m owner--name --vram-budget 22G --ram-budget 12G
```

Private/gated repos: set `HF_TOKEN`. Each model keeps its own hot-expert profile
(`hot-experts.txt` next to it) unless `--profile` is given.

### Repack the experts (recommended, once per model)

```
glm-ssd repack                 # the default model; -m for another one
```

This writes `<model>.glm-ssd-experts` next to the GGUF: every expert as one 4 KiB-aligned block,
so loading an expert from the SSD is one read instead of three. It takes ~95 GB of disk and about
a minute; decode gets ~11% faster and a fresh prompt's first token arrives sooner. The engine
uses the file automatically while it matches the model (the start-up line says
`experts from the repacked file`). If the file is deleted, or stale because the model changed,
the engine reads the GGUF as before; run `glm-ssd repack` again to recreate it (`--force`
rebuilds an up-to-date one).

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
| `-m PATH` | the model: a folder (searched with its subfolders; a folder name inside `models/` is enough) or any shard. Default: the official UD-IQ2_XXS. Also `GLMSSD_MODEL` |
| `--ctx N` | max context (default 32768; 131072 fits on 24 GB) |
| `--chunk N` | prefill tokens per batched step (default 512) |
| `--prefill-batch N` | prefill tokens per pass over the model (default 8192, ~0.8 GB VRAM; 0 = off). Each layer runs over all of them before the next, so the experts not in cache are read from the SSD once per pass instead of once per chunk (a 512-token chunk touches nearly all of them) |
| `--mtp` | MTP speculative decoding |
| `--preload` | fill the expert caches from the learned hot-expert profile at start |
| `--profile FILE` / `--no-profile` | where usage statistics are kept (default `models/hot-experts.txt`; for another `-m` model, `hot-experts.txt` in its folder) |
| `--prefetch K` | prefetch K predicted experts of the next layer while decoding (default 2) |
| `--gguf-experts` | read experts from the GGUF even when a repacked file exists (for comparisons) |
| `--cache-aware M` | cache-aware routing: an uncached expert of rank 3-8 is replaced by a cached one whose routing score is within M (default 0 = exact). 0.1 = +65% decode speed at unchanged perplexity on our test; 0.2 = +95% at +1.3% perplexity |
| `--kv-cache q8` | store the context (MLA latents) as Q8: 47% smaller (~0.85 GB saved at 128K), no measurable quality change |
| `--idle-unload SECS` | chat/serve: after this many idle seconds, free all VRAM and RAM (incl. the CUDA context); the next message/request reloads in ~5 s (default 120, 0 = never). The conversation's prompt cache is parked in a temp file meanwhile, so the conversation continues without being re-read |
| `--no-reasoning-replay` | serve: when a client sends an earlier answer back without its `reasoning_content` (most OpenAI clients do), the server normally puts the reasoning back, so the conversation stays identical to what the model processed and its prompt cache is reused. This flag renders the history exactly as the client sent it (the earlier answer is re-read: slower) |
| `--max-output N` | chat/serve: max tokens per answer, reasoning included (default 0 = no limit: the answer runs until the model stops or the context window is full). In serve, a request's own `max_tokens` takes precedence |
| `--snapshots N` | prompt-cache snapshots kept in RAM (~150 MB each) |
| `--temp --top-p --top-k --min-p --seed` | sampling |

Chat commands: `/effort low|high|max`, `/think` (toggle reasoning display), `/reset`, `/stats`, `/exit`.

Reasoning effort: GLM-5.3 has three levels, `low`, `high` and `max` (the default), and always
reasons; there is no "off". OpenAI-style values are mapped: `none`/`minimal` → `low`,
`medium` → `high`, `xhigh` → `max`. The server reads `reasoning_effort` or `reasoning.effort`.

The server reports its context window (`--ctx`) in `GET /v1/models` (`context_length`,
`max_model_len`, `max_context_length`), `GET /health` and llama.cpp-style `GET /props` (`n_ctx`).
A prompt that does not fit gets HTTP 400 `context_length_exceeded`.

The server supports streaming (`reasoning_content` and `content` deltas), tools (OpenAI format,
mapped to GLM's `<tool_call>` XML), `reasoning_effort`, `stop`, and `stream_options.include_usage`.
Requests are served one at a time; consecutive requests that share a prefix (multi-turn chats,
agent loops) reuse it: only the new tokens are processed (`usage.prompt_tokens_details.cached_tokens`).
For that, the client must resend the earlier parts of a chat unchanged: anything that changes
near the start (e.g. the current time in the system prompt, or a different `reasoning_effort`)
makes the whole conversation be read again. The server logs where a request stops matching the
cached conversation (`(prompt cache: request differs ... at token N ...)`) with the text on both
sides. Put per-turn information such as the time in the newest user message instead.

Time to first token is the prefill of the new tokens. Every token routes to 8 of 288 experts
per layer, so even a short new message reads several GB from the SSD (~2 s for a 10-token
follow-up), and a fresh prompt of more than ~100 tokens reads nearly every expert not in cache
once (~10-15 s), plus the compute for its length. If your client hides `reasoning_content`, the
wait you see also includes the thinking: use `reasoning_effort` low or high for snappier replies.

Diagnostics: `glm-ssd ppl --file text.txt` (perplexity + speed in decode mode, to compare
options), `inspect`, `qtest`, `qbench`, `iobench`, `tokenize`, `template`, `run --trace`
(compare with `llama-eval-callback`), `ref-l0` (f64 CPU reference of layer 0).

## Performance (RTX 3090 + Samsung 990 PRO, 22G / 12G budgets)

| | |
|---|---|
Measured with repacked experts (`glm-ssd repack`), `--chunk 1024 --kv-cache q8 --ctx 65536`, cold
caches, 300 generated tokens:

| | |
|---|---|
| decode, exact routing, `--mtp` | ~6.1 tok/s |
| decode, `--mtp --cache-aware 0.1` | ~10 tok/s (no measurable perplexity change on our texts) |
| decode, `--mtp --cache-aware 0.2` | ~12 tok/s (~+2% perplexity) |
| prefill, fresh 2.1K-token prompt | 18.6 s (~113 tok/s) |

Decode is bound by SSD reads of the experts not in cache (~0.4 GB per token, the drive at its
random-read maximum while busy); a larger `--ram-budget` / `--vram-budget` is the main lever.
`glm-ssd run` prints a per-token breakdown (router sync, SSD wait, launches) and SSD utilization.
