"""
Monotonic Alignment Search (MAS) wrapper with Cython fast path and
exact NumPy Viterbi fallback when Cython extension is not compiled.
"""

from typing import Optional
import numpy as np
import torch

_maximum_path_c = None
try:
    from .monotonic_align.core import maximum_path_c as _maximum_path_c  # type: ignore
except ImportError:
    try:
        from .core import maximum_path_c as _maximum_path_c  # type: ignore
    except ImportError:
        _maximum_path_c = None


def _maximum_path_each_numpy(
    path: np.ndarray,
    value: np.ndarray,
    t_y: int,
    t_x: int,
    max_neg_val: float = -1e9,
) -> None:
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute optimal monotonic alignment path.

    Args:
        neg_cent: [b, t_t, t_s] log-likelihood matrix.
        mask: [b, t_t, t_s] binary attention mask.
    Returns:
        Binary alignment tensor of shape [b, t_t, t_s].
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent_np = np.ascontiguousarray(neg_cent.detach().cpu().numpy().astype(np.float32))
    path = np.zeros(neg_cent_np.shape, dtype=np.int32)

    t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].detach().cpu().numpy().astype(np.int32))
    t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].detach().cpu().numpy().astype(np.int32))

    if _maximum_path_c is not None:
        _maximum_path_c(path, neg_cent_np, t_t_max, t_s_max)
    else:
        for i in range(path.shape[0]):
            _maximum_path_each_numpy(path[i], neg_cent_np[i], int(t_t_max[i]), int(t_s_max[i]))

    return torch.from_numpy(path).to(device=device, dtype=dtype)
