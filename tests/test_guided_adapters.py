#!/usr/bin/env python
"""Tests for the model adapters and the cross-vocabulary token mapper.

    python tests/test_guided_adapters.py

Covers two things that are easy to break silently:

* The Dream call path must be unchanged. The adapter has to issue exactly the
  positional/keyword call the decode loop used before, and Dream's logits must
  still be shifted by one position while LLaDA's are not.
* Cross-vocabulary conversion must preserve length. Verification indexes drafts
  and verifier logits by the same position and returns ``accept_len`` as a count
  of draft tokens, so a conversion that changes length misaligns the sequence.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guided_diffusion.guided_diff_utils import (
    DreamModelAdapter,
    LLaDAModelAdapter,
    TokenMapper,
    _draft_logits,
)


class RecordingModel:
    """Captures how the adapter called it."""

    def __init__(self):
        self.calls = []
        self.config = type("cfg", (), {"num_hidden_layers": 2})()
        layer = type("layer", (), {"reset_block_cache": lambda s, *a: self.resets.append(a)})()
        self.resets = []
        self.model = type("inner", (), {"layers": [layer, layer]})()

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "out"

    def reset_block_cache(self, *args):
        self.resets.append(args)


class FakeTokenizer:
    """Minimal tokenizer over single-character tokens."""

    def __init__(self, chars, offset):
        self.chars = list(chars)
        self.offset = offset
        self.unk_token_id = 0
        self.vocab_size = len(self.chars)

    def __len__(self):
        return len(self.chars)

    def batch_decode(self, batches, skip_special_tokens=False):
        return ["".join(self.chars[i] for i in b) for b in batches]

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            out.append([self.chars.index(c) for c in t if c in self.chars])
        return type("enc", (), {"input_ids": out})()

    def get_vocab(self):
        return {c: i for i, c in enumerate(self.chars)}

    @property
    def special_tokens_map(self):
        return {}


def main() -> int:
    failures = []

    # ── Dream call convention is preserved ────────────────────────────────
    m = RecordingModel()
    DreamModelAdapter.forward(
        m, "SEQ", "POS", max_length=512, block_size=64, save_cache=True, clean_idx=7
    )
    args, kwargs = m.calls[0]
    expected_args = ("SEQ", None, "POS")
    expected_kwargs = {
        "use_block_diffusion": True,
        "use_full_query_attn": False,
        "max_length": 512,
        "block_size": 64,
        "save_cache": True,
        "clean_idx": 7,
    }
    if args != expected_args:
        failures.append(f"Dream positional args changed: {args} != {expected_args}")
    if kwargs != expected_kwargs:
        failures.append(f"Dream kwargs changed: {kwargs} != {expected_kwargs}")
    print(f"[1] Dream forward call convention unchanged : {args == expected_args and kwargs == expected_kwargs}")

    DreamModelAdapter.reset_block_cache(m, 1, 512, 64)
    if len(m.resets) != 2:
        failures.append(f"Dream cache reset should touch 2 layers, touched {len(m.resets)}")
    print(f"[2] Dream resets every layer               : {len(m.resets)} layers")

    # ── LLaDA uses the model-level reset and no logit shift ───────────────
    m2 = RecordingModel()
    LLaDAModelAdapter.forward(
        m2, "SEQ", "POS", max_length=512, block_size=64, save_cache=False, clean_idx=None
    )
    _, kw = m2.calls[0]
    if kw.get("use_full_query_attn") is not None:
        failures.append("LLaDA adapter must not pass Dream-only use_full_query_attn")
    if kw.get("position_ids") != "POS" or kw.get("use_block_diffusion") is not True:
        failures.append(f"LLaDA kwargs wrong: {kw}")
    print("[3] LLaDA forward call convention          : ok")

    # ── the logit shift applies to Dream only ─────────────────────────────
    logits = torch.arange(12, dtype=torch.float).view(1, 4, 3)
    shifted = _draft_logits(logits, DreamModelAdapter)
    unshifted = _draft_logits(logits, LLaDAModelAdapter)
    if not torch.equal(shifted[:, 1:], logits[:, :-1]):
        failures.append("Dream logits are not shifted right by one position")
    if not torch.equal(unshifted, logits):
        failures.append("LLaDA logits must not be shifted")
    print("[4] logit shift: Dream yes / LLaDA no      : ok")

    # ── cross-vocabulary conversion preserves length ──────────────────────
    # Two different vocabularies over the same characters, in different orders,
    # so IDs genuinely differ between the two sides.
    src = FakeTokenizer("abcdef", 0)
    dst = FakeTokenizer("fedcbaxyz", 0)  # different order AND different size
    mapper = TokenMapper(src, dst)
    if mapper.skip_conversion:
        failures.append("test setup: expected skip_conversion to be False")

    ids = torch.tensor([0, 1, 2, 3, 4, 5, 2, 1])
    conv = mapper.dream_to_qwen_ids(ids)
    if conv.shape != ids.shape:
        failures.append(f"conversion changed length: {ids.shape} -> {conv.shape}")
    back = mapper.qwen_to_dream_ids(conv)
    if not torch.equal(back, ids):
        failures.append(f"round trip lost information: {ids.tolist()} -> {back.tolist()}")
    print(f"[5] cross-vocab conversion is 1:1 in length: {tuple(ids.shape)} -> {tuple(conv.shape)}")
    print(f"[6] round trip through both vocabs         : {back.tolist() == ids.tolist()}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll adapter / token-mapper checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
