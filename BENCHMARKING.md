# Benchmarking this repo as a baseline (GSM8K, generation length 512)

This document describes how to run the guided-diffusion method in this repository
as a **baseline**, reporting the three metrics used in dLLM acceleration papers:

1. **Accuracy** — exact-match on the GSM8K final answer
2. **Throughput** — decoding tokens per second
3. **Tokens per step** — tokens committed per denoising step

## What this repo implements

The evaluator drives *AR-assisted guided diffusion*: a Dream diffusion model
proposes tokens, and a small autoregressive model (Qwen2.5-1.5B-Instruct)
decides which proposals are safe to unmask. It is a **two-model** method with a
sliding-window KV cache.

Two things to know before you compare numbers:

- **Dream and LLaDA.** `guided_diffusion/` was originally written against the
  Dream modeling code only. LLaDA support is added here via
  `src/model/llada_flash/` plus a model adapter — see "Running LLaDA" below.
  The upstream release shipped no cached LLaDA implementation: the modules
  `src/evaluator/llada_eval.py` imports (`block_cached_llada`, `opt_llada`,
  `llada_v2`) are absent, so that evaluator still raises `ImportError`. It is
  unused by the benchmark path.

### Two pre-existing bugs fixed here

Both block a fresh checkout from running at all, independently of LLaDA:

- `src/model/dream_flash/generation_utils.py` had a dangling `else:` at line
  1064 — commit 552eaae commented out the `if` branch above it and left the
  `else` behind. That is a `SyntaxError`, so importing `src.model.auto_map`
  failed and no evaluator could start. The block is now unconditional, which is
  what removing that branch intended.
- `requirements_minimal.txt` pinned `transformers>=4.56.0`, but
  `guided_diffusion` imports `transformers.generation.utils._crop_past_key_values`
  (live code, `guided_diff_utils.py:695`), which was removed in 4.55. Installing
  per the old pin failed at import. Now pinned `>=4.51.0,<4.54`; verified on
  4.53.3.

One thing left alone: `src/model/dream_flash/generation_utils.py:35` calls
`AutoTokenizer.from_pretrained` at module import, so importing the package
requires network access or a warm HF cache. Harmless in a normal setup.
- **`Dream-org/Dream-Flash-Instruct-7B` is not a separate checkpoint.**
  `src/model/auto_map.py` rewrites `-Flash` to `-v0`, so it loads
  `Dream-org/Dream-v0-Instruct-7B` weights under this repo's modeling code. The
  weights are stock Dream-7B-Instruct.

## Running it

```bash
bash scripts/run_gsm8k_512_benchmark.sh
```

That wraps:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
python guided_diffusion/dream_eval/gsm8k_guided_evaluator.py \
  --config test_configs/dream/gsm8k/guided_diffusion/gsm8k_512_5shot_paper_protocol.yaml
```

### Choosing an acceptance rule

`dream.sampling_strategy` selects how the AR guider decides a draft token is
safe to unmask. The mapping to the "Top-N Match" rows the FlashDLM paper
reports:

| Paper row | Config | Provided as |
| --- | --- | --- |
| Top-1 Match | `sampling_strategy: deterministic` | `gsm8k_512_5shot_paper_protocol_deterministic.yaml` |
| Top-2 Match | `sampling_strategy: topk`, `acceptance_top_k: 2` | `gsm8k_512_5shot_top2match.yaml` |
| Top-5 Match | `sampling_strategy: topk`, `acceptance_top_k: 5` | `gsm8k_512_5shot_top5match.yaml` |
| *(not published)* | `sampling_strategy: topk_relative` | `gsm8k_512_5shot_paper_protocol.yaml` |

Note the last row: the config shipped in this repo uses `topk_relative`, which
adds a relative-probability threshold (`top_p`, read as `relative_threshold`)
on top of top-k membership. It does not correspond to any published row. Use
one of the first three if you want a number that lines up with the paper.

### Shot count

The FlashDLM paper reports GSM8K at **8-shot**. If your comparison table is
5-shot, you need `nshot: 5` (what these configs use) and the two protocols are
not interchangeable. To validate a setup against the paper's published numbers
first, set `nshot: 8`, confirm you land near their figures, then switch to 5.

Results land in `results/baselines/flashdlm/gsm8k_512_5shot/<node>_<gpu>/`:

- `evaluation.log` — per-sample trace plus a `FINAL EVALUATION SUMMARY` block
- `eval_results.json` — per-sample records (latency, generated length,
  denoising steps, AR calls, correctness)

To recompute the aggregate metrics from the JSON at any time:

```bash
python scripts/summarize_flashdlm_metrics.py \
  results/baselines/flashdlm/gsm8k_512_5shot/*/eval_results.json
```

### Resuming

The evaluator **skips a run entirely** if `evaluation.log` already contains
`FINAL EVALUATION SUMMARY`. Delete the run directory to force a re-run.

## Metric definitions

All three are aggregates over the run, not means of per-sample ratios:

| Metric | Definition |
| --- | --- |
| Accuracy | correct samples / 1319 |
| Throughput | Σ generated tokens / Σ generation wall-clock seconds |
| Tokens per step | Σ generated tokens / Σ denoising steps |

`generated tokens` counts everything emitted up to and including the first EOS.
The wall-clock timer wraps the whole `generate()` call — prefill included — and
is bracketed by `torch.cuda.synchronize()`.

If you define throughput as decode-only (excluding prefill), subtract the first
step's time from `time_breakdown` in `eval_results.json`; at 512 tokens with a
5-shot prompt the difference is small but not zero, so state which convention
you used.

## Settings that change the numbers

| Setting | Where | Why it matters |
| --- | --- | --- |
| `model.single_device: true` | config | With 2+ visible GPUs the evaluator puts the diffusion model on `cuda:0` and the AR model on `cuda:1`, adding a host-mediated copy to every verification step. Benchmark on one GPU. |
| `model.dtype` | config | Model loading defaults to `float16`; the configs here set `bfloat16` to match the usual dLLM evaluation setup. |
| `eval.warmup_samples` | config | One untimed generation absorbs lazy CUDA init and kernel autotuning that would otherwise be charged to sample 1. |
| `dataset.nshot` | config | Controls the number of CoT exemplars. Set to 5 for a 5-shot GSM8K comparison. |
| `dream.block_size`, `dream.sliding_window_size` | config | Cache geometry. `64` / `(256, 256)` are this repo's defaults. |
| `dream.early_stop` | config | Stops once EOS is decoded. Leave on — off, every sample pays the full 512-token budget and throughput is not comparable. |

## Notes on previously reported numbers

Three fixes in this branch affect metrics produced by earlier runs of this code:

- **`nshot` was ignored.** `GSM8K.wrap_cot()` always emitted all 8 exemplars
  regardless of the config value, so any run labelled 5-shot was actually
  8-shot. Fixed in `src/dataset/gsm8k.py`.
- **Tokens per step double-counted EOS.** The summary used
  `(total_tokens + num_samples) / total_steps`, but `generated_length` already
  includes the EOS token. On GSM8K-512 the inflation is roughly
  `num_samples / total_steps`. Fixed in `base_guided_evaluator.py`.
- **Timing was not synchronized.** The latency timer now brackets the generation
  call with `torch.cuda.synchronize()`.

Per-sample `denoising_steps` and `ar_model_calls` are now written to
`eval_results.json`; older result files lack these fields and the summarizer
will report tokens-per-step as `n/a` for them.

## Running LLaDA

Guided diffusion runs on LLaDA-8B-Instruct and LLaDA-1.5 through the same
evaluator; only the config changes.

```bash
bash scripts/run_gsm8k_512_benchmark.sh \
  test_configs/llada/gsm8k/guided_diffusion/gsm8k_512_5shot_llada8b_top2match.yaml
```

Configs provided:

| Config | Model | Protocol |
| --- | --- | --- |
| `gsm8k_512_5shot_llada8b_top2match.yaml` | LLaDA-8B-Instruct | 5-shot, Top-2 Match |
| `gsm8k_512_5shot_llada15_top2match.yaml` | LLaDA-1.5 | 5-shot, Top-2 Match |
| `gsm8k_512_8shot_llada8b_top1match_validation.yaml` | LLaDA-8B-Instruct | 8-shot, Top-1 Match — reproduces the paper's published setting |

LLaDA-1.5 is the same architecture as LLaDA-8B-Instruct (`model_type: llada`,
`LLaDAModelLM`), so it loads under the same implementation. Confirm the HF repo
id in the config before running.

### How it works

* `src/model/llada_flash/` is the block-cached implementation. `src/model/llada/`
  is the stock upstream code and is left byte-identical — that is what you run
  for the uncached 1.0x baseline row.
* `model.family` in the config (`dream` or `llada`) picks a model adapter in
  `guided_diff_utils.py`, which handles the two things that differ between
  architectures: how the block cache is reset, and whether logits need Dream's
  one-position shift. LLaDA is a mask predictor — position *i* predicts token
  *i* in place — so no shift is applied. Defaulting `family` to `dream` keeps
  every existing Dream config on exactly the previous code path.
* Cross-vocabulary conversion between the drafter and the Qwen guider goes
  through a precomputed per-ID table so it is 1:1 in length. This matters:
  verification indexes drafts and verifier logits by position and reports
  `accept_len` as a count of draft tokens, so a conversion that changes length
  silently misaligns the sequence. The previous bulk `decode()`→`encode()` did
  change length; it was unreachable for Dream, which shares Qwen's tokenizer.

### Tests

Both run on CPU in seconds with small random models — no checkpoint download:

```bash
python tests/test_llada_block_cache.py   # cache == full recompute, MHA and GQA
python tests/test_guided_adapters.py     # Dream call path unchanged; 1:1 conversion
```

`test_llada_block_cache.py` checks that a full-length block-diffusion forward
equals the stock LLaDA forward, and that a windowed forward equals the stock
forward restricted to that window (positions outside the window served from
cache). Both match exactly. A negative control confirms the windowed check is
not vacuous.

These tests do not cover end-to-end generation quality on real weights. Run the
8-shot validation config against the published numbers below before trusting a
LLaDA accuracy or throughput figure.

### Validating against published numbers

The published LLaDA figures give you something to check against. Useful
reference points, all GSM8K 8-shot (generation length is not stated in those
tables, so match it before comparing):

| Setting | Accuracy | Latency |
| --- | --- | --- |
| LLaDA-8B baseline, block length 64 | 79.30 | 56.89 s |
| LLaDA-8B + Qwen2.5-1.5B guider, Top-1 Match | 79.91 | 4.29 s |
| LLaDA-8B + Qwen2.5-3B guider, Top-5 Match | 80.06 | 4.29 s |
| Dream-7B + Qwen2.5-1.5B guider | 80.3 | 2.55 s |

The last two rows give a cheap sanity ratio: with the same Qwen2.5-1.5B guider,
LLaDA guided decoding is roughly 1.7x the per-sample latency of Dream guided
decoding (4.29 s vs 2.55 s). A port landing far from that ratio at matching
settings is probably wrong somewhere.

## Related methods not in this repo

FreeDave (draft-then-verify with multiple independent forward passes) is a
separate method with its own implementation — it is not the code in this
repository, and nothing here reproduces it.
