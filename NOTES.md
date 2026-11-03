# Engineering notes / handoff

Working notes that are not obvious from the code. DESIGN.md has the architecture, roadmap and
measurements; this file has *state, gotchas and the exact plan for the next step*.

## Current state (2026-11-02)

- M0-M10 implemented and tested end to end (see DESIGN.md roadmap + results). Usage in README.
- Model: all 4 shards in `models/UD-IQ2_XXS/`, SHA-256 verified against HF (sha256sum -c).
- Reference: llama.cpp master @785322d in `ref/llama.cpp` (gitignored), CUDA build done:
  `ref/llama.cpp/build/bin/Release/llama-eval-callback.exe` (no `--no-warmup` flag).
- Permissions: auto-mode classifier blocks building/running code from `ref/` ("Code from
  External"). The user added allow rules in `.claude/settings.local.json` (gitignored) for:
  `cmake -S ref/llama.cpp *`, `cmake --build ref/llama.cpp/build *`, the three
  `./ref/llama.cpp/build/bin/Release/llama-*.exe *` binaries, and the scratchpad venv python
  running `tools/qref.py`. **Run those commands plainly (no `cd … &&` prefix) so rules match.**
- Machine during testing: ~18 GB RAM free of 32 -> tests used `--ram-budget 12G`.

### How things were tested (rerun after changes)
- `glm-ssd qtest -m <shard2>`: dequant / GEMV / GEMM vs CPU reference, all quant types.
- `glm-ssd run --prompt "[gMASK]<sop><|user|>
What is the capital of France?<|assistant|>
<think></think>"`
  -> "The capital of France is **Paris**." (`--sequential` reproduces the validated M2 logits
  exactly; the batched GEMM path differs ~1% from f16 activation rounding).
- Needle tests via `run --prompt-file` (3.5K and 17K tokens) exercise the sparse DSA path.
- `bench` for decode speed; `chat` with piped stdin; `serve` + curl (stream, tools, tool result
  round trip: 160/201 prompt tokens reused from the snapshot).

### Known limits / ideas
- Decode is SSD-bound: ~0.4 GB of expert reads per token (user flags), SSD busy ~60% of decode
  time at ~5.7 GB/s (its random-read max) while busy; the rest is the serial per-layer chain
  (attention + router -> sync -> misses -> experts). `run` prints the breakdown.
  Done 2026-11-03: repack (one read per expert, +11%), fused MoE kernels (t = 1 and 2), one
  router sync per layer, dense cache maps, wide GEMV for few-row matrices.
  Tried, no gain / rejected: copy kernel instead of cudaMemcpyAsync; deeper, throttled or chunked
  prefetch (more bytes read than saved); dp4a int8-activation K-quant GEMV (+57% Q5_K bandwidth
  but +2% ppl); 2-4 rows per warp for K-quant/IQ GEMVs (instruction-bound, more registers);
  dropping uncached low-weight experts (worse than --cache-aware at equal speed).
  A/B noise: decode runs vary ~±4%, ppl on 300 tokens ~±0.02; --cache-aware outputs depend on
  read timing, so check correctness with exact routing.
- Prefill (layer-major sweep) is SSD-bound for prompts up to a few K tokens: ~70 GB of uncached
  experts per pass. Ideas: prefetch the next layer's hot experts during the current layer's
  MoE; per-expert contiguous repack. Prompts <= one chunk (512) don't use the sweep (no gain).
- TTFT probe used for the prefill / prompt-cache work: a stdlib Python client streaming
  /v1/chat/completions (fresh prompts of 66-3.5K tokens; 4-turn chats with the client keeping vs
  dropping `reasoning_content`), reading TTFT and the server's per-request `prompt N tok (M cached)`.
- Sweep vs chunked logits differ by f16-GEMM rounding noise (same size as --chunk 512 vs 2048,
  and run-to-run variation from expert arrival order); greedy continuations match.
- Speculative prefetch with larger K hurts (wrong guesses steal SSD bandwidth); K=2 is neutral-positive.
- Server handles one request at a time (single sequence engine).
- Expert cache tiers can't hand experts down VRAM -> RAM (single copy engine on this GPU/driver).

## M2 validation recipe (done once; rerun after changing the forward pass)

Results: per-layer sums within 1-3% of llama.cpp; the residual gap is llama.cpp's Q8_1
activation quantization. Arbiter: `glm-ssd ref-l0 --token 785` (f64 CPU, layer 0) matches our
GPU trace to 6 digits. Comparison scripts used (session scratchpad, not in repo): parse
`common_debug_cb_eval: <name> = ... sum = X` from the reference log and `[pos] name sum=` from
ours; for multi-token prompts compare the LAST printed row of each ref tensor with our last
position. llama-eval-callback has no `--no-warmup` flag in this version.

1. `glm-ssd inspect --vram-budget 22G --ram-budget 24G` (full open, checks all shards).
2. Tokenizer check: run llama-eval-callback with a prompt and compare its printed token ids to
   `glm-ssd tokenize "<same text>"`. glm4/glm5 pre-tokenizer: NO automatic BOS.
3. Reference activations: `./ref/llama.cpp/build/bin/Release/llama-eval-callback.exe -m
   models/UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf -p "<text>" -ngl 99 --cpu-moe
   -c 512` (experts via mmap on CPU: slow but fine for a few tokens; RAM is 32 GB). Output is
   huge -> redirect to a file in the scratchpad and grep tensor names.
4. Ours: `glm-ssd run --tokens <ids> --trace -n 4` (same ids). Trace names mirror llama.cpp
   callback names: `attn_norm-N, kda_scan_out-N, kqv_out-N, attn_out-N, ffn_norm-N,
   ffn_moe_out-N, ffn_shexp-N, ffn_out-N, l_out-N, result_norm, result_output`.
   llama.cpp sums over ALL prompt tokens at once; ours prints per token -> start with a
   1-token prompt (then sums are directly comparable), then compare the last-token logits /
   greedy continuation for a longer prompt.
5. The DSA sparse path only activates beyond 2048 tokens (512 pools x 4): needs a separate
   long-prompt test later.

### If activations diverge, check these first (semantics from transformers modeling_glm5_next.py)
- mHC: mixes = hc_fn @ rmsnorm_unweighted(flatten 4x4096, eps=rms_norm_eps 1e-5).
  pre = sigmoid(m[0:4]*scale0+base)+hc_eps; post = 2*sigmoid(m[4:8]*scale1+base);
  comb[i][j] = m[8+4i+j]*scale2+base, softmax over j, +eps, then COLUMN norm first (sum over i),
  then (iters-1) x (row norm, col norm). Output stream d = post[d]*out + sum_s comb[s][d]*res[s].
  hc_eps = 1e-6 (GGUF hyper_connection.epsilon). Final head = plain mean of the 4 streams.
- KDA: conv tap w[0] multiplies the oldest input; SiLU after conv; q,k l2norm eps 1e-6
  (x/sqrt(sum+eps)); q *= 1/sqrt(128). GGUF `ssm_a` = -exp(A_log) (llama.cpp convention), so
  g = lb * sigmoid(-ssm_a * (f + dt_bias)), lb = -5. beta = sigmoid(ssm_beta x).
  State update per head: S *= exp(g) (row k), kv = S^T k, delta = (v - kv)*beta, S += k delta^T,
  o = S^T q. Output: rmsnorm(o, eps 1e-5) * w * sigmoid(g_b(g_a x)) per head, then o_proj.
- MLA (nope only, no RoPE): latent = rmsnorm(kv_a x) (512) cached; q = q_b(rmsnorm(q_a x));
  `attn_k_b` dims [256,512,64]: per head rows=512, K=256 -> q_abs = Wkb q; scores scaled
  1/sqrt(256); o_lat = softmax . latents; `attn_v_b` [512,256,64]: per head rows=256, K=512.
- Indexer: k = LayerNorm(wk x, eps 1e-6, affine); gate = compressor_gate x; pooled key per
  complete pool of 4 = per-channel softmax over members of (gate+ape) weighting keys;
  score = sum_h w_h relu(q_h.P/sqrt(128)) (the w * 32^-0.5 factor doesn't change ranking);
  top 512 pools + incomplete tail tokens. Weights proj input = attention input (normed x).
- MoE: scores = sigmoid(router logits) (F32 router); choose top-8 by score + exp_probs_b;
  weights = chosen scores / (sum + 1e-20) * 2.5. SwiGLU clamp: silu(min(g,10)) * clamp(u,±10);
  applies to routed experts (clamp_exp), shared expert AND dense layers 0-2 (clamp_shexp).
- Expert tensor slices: `[in, out, 288]`, expert e = contiguous bytes n_bytes/288 * e.

## Build / tooling gotchas (cost time once already)
- `cc` adds `-G` (device debug, no optimization: 60x slower kernels) when the cargo profile has
  debug info -> `b.debug(false)` in crates/cuda/build.rs. Keep it.
- nvcc needs cl.exe: build.rs finds it with vswhere and sets `NVCC_CCBIN`.
- Kernels use C++20 templated lambdas (`-std=c++20`).
- In GEMV decoders never convert 64-bit ints to float (I2F.U64 is a slow multi-op); use the
  `u8f`/`s8f` exponent trick. IQ tables live in shared memory, GEMV blocks are persistent.
- Editing source with Python heredocs mangles `\r\n`, `\p`, `\0` escapes -> prefer the Edit
  tool; Python on Windows also writes CRLF (harmless for Rust).
- Nsight Compute: use `.../target/windows-desktop-win7-x64/ncu.exe` directly (the .bat breaks
  args), `--kernel-name-base demangled --kernel-name "regex:QIQ2_XXS, \(int\)1>"`.

## Engine invariants worth remembering
- `Engine::pos` == number of tokens whose state is in the caches; `Session::history` mirrors it.
- MTP: `mtp_pos` is the next MTP position; exactly one "pending" row (hidden of the last trunk
  position, waiting for its next token) exists after a prefill or a finished generation.
- Snapshots are invalidated when a request diverges before them (positions get overwritten).
  One is taken at the end of EVERY prompt: without it a client that changes the last answer
  (stripped reasoning, whitespace) forced a re-read of the whole conversation (28 s at 835 tokens).
- Saved session (idle unload) = snapshot at pos + MLA latents/scales/pooled keys up to
  max(pos, mtp_pos) + Session history/logits/snapshots; only valid for the same engine options.
- Expert slot reuse is guarded by events: RAM slot -> its last H2D; VRAM slot -> its last use.
- Each MoE size class (7.19 / 8.38 / 8.69 / 9.38 MiB) has its own slot pools (>= 24 slots each).
