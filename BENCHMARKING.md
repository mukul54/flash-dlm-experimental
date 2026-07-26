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

- **Dream only.** `guided_diffusion/` is written against the Dream modeling code
  (`src/model/dream_flash/`). There is no LLaDA path, so results from this repo
  belong in a Dream-7B comparison, not a LLaDA-1.5 one. Porting to LLaDA means
  reimplementing the block/sliding-window cache hooks
  (`reset_block_cache`, etc.) in `src/model/llada/modeling_llada.py`.
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

A second config,
`gsm8k_512_5shot_paper_protocol_deterministic.yaml`, runs the stricter
exact-match acceptance rule instead of top-2/relative-0.5. Report whichever
setting you describe in the text; the deterministic one is more conservative on
throughput and slightly better on accuracy.

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

## Related methods not in this repo

FreeDave (draft-then-verify with multiple independent forward passes) is a
separate method with its own implementation — it is not the code in this
repository, and nothing here reproduces it.
