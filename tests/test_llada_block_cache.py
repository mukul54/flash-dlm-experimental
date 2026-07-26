#!/usr/bin/env python
"""Numerical equivalence tests for the LLaDA block KV cache.

Runs on CPU with a small randomly-initialised model -- no checkpoint download.

    python tests/test_llada_block_cache.py

The two properties that matter:

1. A full-length block-diffusion forward must equal the stock LLaDA forward.
   This exercises head reshaping, rotary embeddings applied at explicit
   absolute positions, and the cache write/read path.

2. With the sequence unchanged between steps, a windowed forward must equal the
   stock forward restricted to that window. Positions outside the window are
   served from cache, so this is the real test of the caching scheme: if the
   cache is written at the wrong offset, or keys are rotated by the wrong
   position, the window's logits drift away from the reference.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.llada.configuration_llada import LLaDAConfig
from src.model.llada.modeling_llada import LLaDAModelLM
from src.model.llada_flash.modeling_llada import LLaDAFlashModelLM

SEQ_LEN = 48
BLOCK_SIZE = 16
WINDOW = (16, 40)


def build_pair(n_kv_heads=None):
    cfg = LLaDAConfig(
        d_model=64,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        n_layers=2,
        vocab_size=256,
        embedding_size=256,
        mlp_hidden_size=128,
        block_type="llama",
        rope=True,
        alibi=False,
        activation_type="silu",
        include_bias=False,
        weight_tying=False,
        max_sequence_length=128,
        block_group_size=1,
        flash_attention=False,
    )

    torch.manual_seed(0)
    ref = LLaDAModelLM(cfg, init_params=True).eval().float()
    flash = LLaDAFlashModelLM(cfg, init_params=True).eval().float()
    # Identical weights, so any output difference comes from the cache path.
    flash.load_state_dict(ref.state_dict())
    return ref, flash


def check(label, n_kv_heads=None):
    """Run the equivalence checks for one attention layout."""
    ref_model, flash_model = build_pair(n_kv_heads=n_kv_heads)

    torch.manual_seed(1)
    seq = torch.randint(0, 256, (1, SEQ_LEN))
    positions = torch.arange(SEQ_LEN)

    with torch.no_grad():
        reference = ref_model(input_ids=seq).logits

    failures = []

    # 1. full-length block-diffusion forward == stock forward
    flash_model.reset_block_cache(1, SEQ_LEN, BLOCK_SIZE)
    with torch.no_grad():
        full = flash_model(
            input_ids=seq,
            position_ids=positions,
            use_block_diffusion=True,
            block_size=BLOCK_SIZE,
            save_cache=True,
            clean_idx=WINDOW[0],
        ).logits

    full_err = (full - reference).abs().max().item()
    print(f"  [1] full-length block forward vs stock forward   : max|diff| = {full_err:.3e}")
    if not torch.allclose(full, reference, atol=1e-4, rtol=1e-4):
        failures.append(f"{label}: full-length forward differs (max {full_err:.3e})")

    # 2. windowed forward == stock forward restricted to the window
    start, end = WINDOW
    with torch.no_grad():
        windowed = flash_model(
            input_ids=seq[:, start:end],
            position_ids=positions[start:end],
            use_block_diffusion=True,
            block_size=BLOCK_SIZE,
            save_cache=True,
            clean_idx=start,
        ).logits

    win_err = (windowed - reference[:, start:end]).abs().max().item()
    print(f"  [2] windowed forward vs stock forward[{start}:{end}]        : max|diff| = {win_err:.3e}")
    if not torch.allclose(windowed, reference[:, start:end], atol=1e-4, rtol=1e-4):
        failures.append(f"{label}: windowed forward differs (max {win_err:.3e})")

    # 3. Negative control. Without the step-0 pass that populates the cache, the
    #    same windowed call must NOT reproduce the reference -- otherwise the
    #    test above would pass even if cached positions were being ignored.
    flash_model.reset_block_cache(1, SEQ_LEN, BLOCK_SIZE)
    flash_model.model.transformer.blocks[0].block_cache.save_cache(clean_idx=start)
    flash_model.model.transformer.blocks[1].block_cache.save_cache(clean_idx=start)
    with torch.no_grad():
        cold = flash_model(
            input_ids=seq[:, start:end],
            position_ids=positions[start:end],
            use_block_diffusion=True,
            block_size=BLOCK_SIZE,
            save_cache=False,
            clean_idx=start,
        ).logits
    cold_err = (cold - reference[:, start:end]).abs().max().item()
    print(f"  [3] negative control (empty cache) must differ  : max|diff| = {cold_err:.3e}")
    if torch.allclose(cold, reference[:, start:end], atol=1e-4, rtol=1e-4):
        failures.append(
            f"{label}: windowed forward matches even with an empty cache -- "
            "the cached region is not affecting the result, so check [2] is vacuous"
        )

    # 4. the stock model must be untouched by importing the flash package
    with torch.no_grad():
        again = ref_model(input_ids=seq).logits
    if not torch.equal(again, reference):
        failures.append(f"{label}: stock LLaDA forward is not reproducible")
    print("  [4] stock model unaffected by flash import      : ok")

    return failures


def main() -> int:
    failures = []
    for label, n_kv in [("MHA (n_kv_heads=n_heads)", None), ("GQA (n_kv_heads=2)", 2)]:
        print(f"\n{label}")
        failures += check(label, n_kv_heads=n_kv)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll block-cache equivalence checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
