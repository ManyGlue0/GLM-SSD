# Engineering notes / handoff

Working notes that are not obvious from the code. DESIGN.md has the architecture, roadmap and
measurements; this file has *state, gotchas and the exact plan for the next step*.

## Current state (2026-11-01)

- M0, M1, M2 done (M2 validated 2026-11-01, see DESIGN.md roadmap). Next: M3.
- Model: all 4 shards in `models/UD-IQ2_XXS/`, SHA-256 verified against HF (sha256sum -c).
  (hfdl aborted at 100% on a state-save race, fixed in 20a10ac; the files were already fine.)
- Reference: llama.cpp master @785322d in `ref/llama.cpp` (gitignored), CUDA build done:
  `ref/llama.cpp/build/bin/Release/llama-eval-callback.exe`. (`llama-cli`/`llama-perplexity`
  targets don't exist in this version: cli became `llama-app`; not needed.)
- Permissions: auto-mode classifier blocks building/running code from `ref/` ("Code from
  External"). The user added allow rules in `.claude/settings.local.json` (gitignored) for:
  `cmake -S ref/llama.cpp *`, `cmake --build ref/llama.cpp/build *`, the three
  `./ref/llama.cpp/build/bin/Release/llama-*.exe *` binaries, and the scratchpad venv python
  running `tools/qref.py`. **Run those commands plainly (no `cd … &&` prefix) so rules match.**
- Python venv with numpy+pyyaml (for gguf-py): session scratchpad `qenv/` (path in settings rule).
  `tools/qref.py` already verified: CPU dequant == gguf-py bit-exact (15 vector sets).

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

## Known simplifications in the M2 code (to revisit)
- Prompt processed one token at a time (no batched prefill yet) -> slow prompts.
- MLA latent + indexer caches are f32 (switch to f16 for long contexts).
- MTP layer 45 not loaded (`Weights::load(.., cfg.n_layer)`).
- No expert caching: 8 synchronous parallel reads per MoE layer through the OS file cache.
- Router top-k and indexer top-k on the host (one sync per MoE layer).

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

## M3 notes (next milestone after validation)
- Direct reads: `io::open_direct` (FILE_FLAG_NO_BUFFERING); 4 KiB-aligned windows; 1-2 reader
  threads already saturate the 990 PRO at ~5.5 GB/s for 7 MiB reads (see DESIGN.md table).
- Expert unit sizes: 7.19 MiB (most layers), 8.38 (12, 44), 9.38 (11), 8.69 (45/MTP).
- Plan: exclusive VRAM + pinned-RAM caches keyed by (layer, expert); frequency+recency eviction;
  speculative prefetch by running layer L+1's router on layer L's hidden state; optional
  one-time repack to per-expert contiguous layout (+~15% read throughput).
