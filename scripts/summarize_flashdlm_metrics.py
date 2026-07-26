#!/usr/bin/env python
"""Summarize a guided-diffusion evaluation run into the three benchmark metrics.

Usage:
    python scripts/summarize_flashdlm_metrics.py <run_dir>/eval_results.json [...]

Metric definitions (aggregate, i.e. totals over the whole run rather than a mean
of per-sample ratios -- long generations therefore carry proportionally more
weight, which is what a throughput number should reflect):

    accuracy    = correct samples / total samples
    throughput  = sum(generated tokens) / sum(wall-clock generation seconds)
    tokens/step = sum(generated tokens) / sum(denoising steps)

`generated tokens` counts everything the model emitted up to and including the
first EOS. `wall-clock generation seconds` is the full generate() call including
prefill, measured around a torch.cuda.synchronize().
"""
import argparse
import json
import sys
from pathlib import Path


def summarize(path: Path):
    with open(path) as f:
        rows = json.load(f)

    if not rows:
        raise SystemExit(f"{path}: no samples logged")

    n = len(rows)
    correct = sum(1 for r in rows if r["is_correct"])
    tokens = sum(r["generated_length"] for r in rows)
    seconds = sum(r["latency_ms"] for r in rows) / 1000.0
    steps = sum(r.get("denoising_steps") or 0 for r in rows)
    ar_calls = sum(r.get("ar_model_calls") or 0 for r in rows)

    print(f"\n{path}")
    print("-" * len(str(path)))
    print(f"samples                : {n}")
    print(f"accuracy               : {correct / n * 100:.2f}%  ({correct}/{n})")
    print(f"throughput             : {tokens / seconds:.1f} tok/s")
    if steps:
        print(f"tokens per step        : {tokens / steps:.2f}")
        print(f"avg denoising steps    : {steps / n:.2f} per sample")
    else:
        print("tokens per step        : n/a (no per-sample step counts in this log)")
    if ar_calls:
        print(f"avg AR verifier calls  : {ar_calls / n:.2f} per sample")
    print(f"avg generated tokens   : {tokens / n:.2f} per sample")
    print(f"avg prompt tokens      : {sum(r['input_length'] for r in rows) / n:.2f} per sample")
    print(f"avg latency            : {seconds / n:.3f} s per sample")

    return {
        "path": str(path),
        "samples": n,
        "accuracy": correct / n,
        "throughput_tok_s": tokens / seconds,
        "tokens_per_step": tokens / steps if steps else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path,
                    help="one or more eval_results.json files")
    ap.add_argument("--json", action="store_true",
                    help="also emit the summary as a JSON array on stdout")
    args = ap.parse_args()

    summaries = [summarize(p) for p in args.results]
    if args.json:
        print()
        json.dump(summaries, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
