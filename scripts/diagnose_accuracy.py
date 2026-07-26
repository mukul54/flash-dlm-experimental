#!/usr/bin/env python
"""Isolate what is costing accuracy, on a 200-sample subset.

    python scripts/diagnose_accuracy.py [--samples 200] [--base <config.yaml>]

Runs a 2x2 over the two settings most likely to explain a low GSM8K score, and
prints accuracy / throughput / tokens-per-step side by side:

  acceptance rule  topk_relative (top-k AND prob >= 0.5 * top_prob)
                   topk          (top-k membership only -- strictly more permissive)
  shot count       8-shot (what the paper reports)
                   5-shot

The permissive rule accepts more draft tokens per step, so it buys throughput
and spends accuracy. If the gap between the two rules is large, that is the
knob; if the gap between 5 and 8 shots is large instead, it is the prompt.

Every variant sees the same 200 problems (same subsample seed), so the numbers
are directly comparable. Uses the same subset as any other run with the same
--samples value.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    ("topk_relative", 8),
    ("topk_relative", 5),
    ("topk", 8),
    ("topk", 5),
]


def build_config(base: dict, strategy: str, nshot: int, samples: int, out_dir: Path) -> Path:
    cfg = copy.deepcopy(base)
    cfg["dataset"]["nshot"] = nshot
    cfg["eval"]["max_samples"] = samples
    cfg["eval"]["resume"] = False
    cfg["eval"]["warmup_samples"] = 1
    cfg["dream"]["sampling_strategy"] = strategy
    cfg["dream"]["acceptance_top_k"] = 2
    if strategy == "topk_relative":
        cfg["dream"]["top_p"] = 0.5          # read as relative_threshold
    else:
        cfg["dream"].pop("top_p", None)
    tag = f"{strategy}_{nshot}shot"
    cfg["save"]["run_dir"] = str(out_dir / tag)
    cfg["wandb"]["flag"] = False

    path = out_dir / f"{tag}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def summarize(run_dir: Path):
    files = list(run_dir.rglob("eval_results.json"))
    if not files:
        return None
    rows = json.loads(files[0].read_text())
    if not rows:
        return None
    n = len(rows)
    tokens = sum(r["generated_length"] for r in rows)
    seconds = sum(r["latency_ms"] for r in rows) / 1000.0
    steps = sum(r.get("denoising_steps") or 0 for r in rows)
    return {
        "n": n,
        "accuracy": sum(1 for r in rows if r["is_correct"]) / n * 100,
        "throughput": tokens / seconds if seconds else 0.0,
        "tokens_per_step": tokens / steps if steps else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--base", type=Path,
                    default=ROOT / "test_configs/dream/gsm8k/guided_diffusion/gsm8k_512_5shot_top2match.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "results/diagnostics")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.base))
    print(f"base config : {args.base}")
    print(f"samples     : {args.samples} (same subset for every variant)\n")

    results = {}
    for strategy, nshot in VARIANTS:
        tag = f"{strategy}_{nshot}shot"
        cfg_path = build_config(base, strategy, nshot, args.samples, args.out)
        print(f"=== running {tag} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, "guided_diffusion/dream_eval/gsm8k_guided_evaluator.py",
             "--config", str(cfg_path)],
            cwd=ROOT,
        )
        if proc.returncode != 0:
            print(f"  {tag}: evaluator exited {proc.returncode}")
        results[tag] = summarize(args.out / tag)

    print(f"\n{'variant':<24} {'n':>5} {'acc %':>8} {'tok/s':>9} {'tok/step':>9}")
    print("-" * 58)
    for strategy, nshot in VARIANTS:
        tag = f"{strategy}_{nshot}shot"
        r = results.get(tag)
        if r is None:
            print(f"{tag:<24} {'--':>5} {'no results':>8}")
            continue
        print(f"{tag:<24} {r['n']:>5} {r['accuracy']:>8.2f} "
              f"{r['throughput']:>9.1f} {r['tokens_per_step']:>9.2f}")

    print("\nRead it as: compare rows down a column to see which knob moves accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
