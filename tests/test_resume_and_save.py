#!/usr/bin/env python
"""Tests for crash-resilient checkpointing and resume.

    python tests/test_resume_and_save.py

A long evaluation must not lose its work to one bad write, and must be able to
pick up where an interrupted run stopped. Neither path involves a model, so
both are testable directly.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the logger without dragging in torch/transformers via the package.
import importlib.util

_src = Path(__file__).resolve().parent.parent / "guided_diffusion/dream_eval/base_guided_evaluator.py"
_text = _src.read_text()
# Pull out just the logger class plus the constants it needs.
_start = _text.index("class GuidedDiffusionLogger")
_end = _text.index("class BaseGuidedEvaluator")
_ns = {"__name__": "logger_under_test"}
exec(  # noqa: S102 - deliberately loading one class in isolation
    "import logging, time\nfrom pathlib import Path\nRED=''\nRESET=''\n" + _text[_start:_end],
    _ns,
)
GuidedDiffusionLogger = _ns["GuidedDiffusionLogger"]


def make_record(logger, idx, correct=True):
    logger.log(
        idx=idx, sample_id=1, steps=None, max_new_tokens=512,
        latency_ms=100.0 + idx, input_length=800, generated_length=50 + idx,
        prompt=f"p{idx}", generated_text=f"g{idx}", ground_truth=f"t{idx}",
        is_correct=correct, time_breakdown={},
        denoising_steps=10 + idx, ar_model_calls=5,
    )


def main() -> int:
    failures = []

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "eval_results.json"

        # ── a normal save round-trips ─────────────────────────────────────
        lg = GuidedDiffusionLogger(str(path))
        for i in range(5):
            make_record(lg, i, correct=(i % 2 == 0))
        if not lg.save():
            failures.append("save() reported failure on a writable path")
        loaded = json.loads(path.read_text())
        if len(loaded) != 5:
            failures.append(f"expected 5 records on disk, found {len(loaded)}")
        print(f"[1] save round-trip                        : {len(loaded)} records")

        # ── no .tmp file is left behind ───────────────────────────────────
        leftovers = list(Path(d).glob("*.tmp"))
        if leftovers:
            failures.append(f"temp files left behind: {leftovers}")
        print(f"[2] no temp file left behind               : {not leftovers}")

        # ── a failing write must not raise, and must not corrupt the file ──
        broken = GuidedDiffusionLogger(str(Path(d) / "nonexistent-dir" / "x.json"))
        make_record(broken, 0)
        try:
            ok = broken.save()
        except OSError:
            ok = None
            failures.append("save() raised on an unwritable path instead of warning")
        if ok is not False and ok is not None:
            failures.append("save() reported success on an unwritable path")
        print(f"[3] unwritable path warns, does not raise  : {ok is False}")

        # the good file must be untouched by the failed write elsewhere
        if len(json.loads(path.read_text())) != 5:
            failures.append("a failed save corrupted an unrelated results file")

        # ── resume reads back prior results ───────────────────────────────
        fresh = GuidedDiffusionLogger(str(path))
        n = fresh.load_existing()
        if n != 5:
            failures.append(f"load_existing returned {n}, expected 5")
        print(f"[4] load_existing recovers prior results   : {n} records")

        # ── counter replay matches a straight-through run ─────────────────
        # This is the bookkeeping resume depends on: totals rebuilt from the
        # file must equal totals accumulated live.
        live = {"corr": 0, "lat": 0.0, "act": 0, "inp": 0, "steps": 0}
        for r in loaded:
            live["corr"] += int(r["is_correct"])
            live["lat"] += r["latency_ms"]
            live["act"] += r["generated_length"]
            live["inp"] += r["input_length"]
            live["steps"] += r.get("denoising_steps") or 0
        replay = {"corr": 0, "lat": 0.0, "act": 0, "inp": 0, "steps": 0}
        for r in fresh.results:
            replay["corr"] += int(r["is_correct"])
            replay["lat"] += r["latency_ms"]
            replay["act"] += r["generated_length"]
            replay["inp"] += r["input_length"]
            replay["steps"] += r.get("denoising_steps") or 0
        if live != replay:
            failures.append(f"counter replay mismatch: {live} vs {replay}")
        print(f"[5] counter replay matches live totals     : {live == replay}")

        # ── appending after resume keeps ordering and count ───────────────
        for i in range(5, 8):
            make_record(fresh, i)
        fresh.save()
        final = json.loads(path.read_text())
        if [r["idx"] for r in final] != list(range(8)):
            failures.append(f"resumed indices out of order: {[r['idx'] for r in final]}")
        print(f"[6] resumed run appends in order           : {[r['idx'] for r in final]}")

        # ── truncated JSON falls back to a clean start ────────────────────
        path.write_text('[{"idx": 0, "latency_ms"')
        salvage = GuidedDiffusionLogger(str(path))
        if salvage.load_existing() != 0:
            failures.append("truncated JSON should resume from 0, not crash or half-load")
        print("[7] truncated results file -> clean restart: ok")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll checkpoint / resume checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
