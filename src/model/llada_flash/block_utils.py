"""Block KV cache for LLaDA block-diffusion decoding.

Mirrors the semantics of ``src/model/dream_flash/block_utils.BlockCache`` but
stores keys and values already split into heads and already rotated, so no
rotary re-application is needed when the cache is read back.

Cache protocol, matching the Dream implementation:

* Step 0 runs a full-length forward. ``update_cache`` writes the whole
  ``[0, max_length)`` range, so every slot holds a real key/value -- there are
  never unwritten zeros participating in attention.
* ``save_cache(clean_idx)`` moves the write cursor to the absolute position
  where the next query window begins.
* Later steps run only a window. ``update_cache`` overwrites
  ``[clean_idx, clean_idx + window_len)``; everything outside the window keeps
  the value it had from an earlier step.
* ``get_cache`` always returns the full ``max_length`` buffers, so each query
  window attends over the entire sequence.
"""

from typing import Optional, Tuple

import torch


class LLaDABlockCache(torch.nn.Module):
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        block_size: int,
        num_key_value_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.block_size = block_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype if dtype is not None else torch.float16

        shape = (batch_size, num_key_value_heads, max_length, head_dim)
        self.key_cache = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.value_cache = torch.zeros(shape, device=self.device, dtype=self.dtype)

        self.clean_cache_idx = 0
        self.next_cache_idx = 0

    def update_cache(self, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """Write rotated keys/values for the current window at the write cursor.

        ``new_k`` / ``new_v`` are ``(batch, n_kv_heads, window_len, head_dim)``.
        """
        start = self.clean_cache_idx
        end = start + new_k.shape[2]
        if end > self.max_length:
            raise ValueError(
                f"block cache overflow: writing [{start}, {end}) into a buffer of "
                f"length {self.max_length}"
            )

        self.key_cache[:, :, start:end] = new_k.to(device=self.device, dtype=self.dtype)
        self.value_cache[:, :, start:end] = new_v.to(device=self.device, dtype=self.dtype)
        self.next_cache_idx = end

    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache, self.value_cache

    def save_cache(self, clean_idx: Optional[int] = None) -> None:
        self.clean_cache_idx = clean_idx if clean_idx is not None else self.next_cache_idx
