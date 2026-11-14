"""
Utilities to construct real Piper `SynthesizerTrn` and `MultiPeriodDiscriminator`
directly from PyTorch Lightning `.ckpt` checkpoints (e.g. `MahtaFetrat/Mana-Persian-Piper`)
without requiring `lightning` or `pysilero_vad` at runtime.
"""

import ast
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch

from .vits_models import MultiPeriodDiscriminator, SynthesizerTrn


def _parse_tuple_param(val: Any) -> Any:
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val


def build_piper_vits_from_checkpoint(
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Tuple[SynthesizerTrn, MultiPeriodDiscriminator, Dict[str, Any]]:
    """
    Loads a Piper Lightning `.ckpt` (`model_g` + `model_d`) into pure PyTorch modules.

    Returns:
        `(model_g, model_d, hparams)`
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp: Dict[str, Any] = dict(ckpt.get("hyper_parameters", {}))

    num_symbols = int(hp.get("num_symbols", 256))
    num_speakers = int(hp.get("num_speakers", 1))
    filter_length = int(hp.get("filter_length", 1024))
    hop_length = int(hp.get("hop_length", 256))
    segment_size = int(hp.get("segment_size", 8192))
    inter_channels = int(hp.get("inter_channels", 192))
    hidden_channels = int(hp.get("hidden_channels", 192))
    filter_channels = int(hp.get("filter_channels", 768))
    n_heads = int(hp.get("n_heads", 2))
    n_layers = int(hp.get("n_layers", 6))
    kernel_size = int(hp.get("kernel_size", 3))
    p_dropout = float(hp.get("p_dropout", 0.1))
    resblock = str(hp.get("resblock", "2"))
    resblock_kernel_sizes = tuple(_parse_tuple_param(hp.get("resblock_kernel_sizes", (3, 5, 7))))
    resblock_dilation_sizes = tuple(
        tuple(x) for x in _parse_tuple_param(hp.get("resblock_dilation_sizes", ((1, 2), (2, 6), (3, 12))))
    )
    upsample_rates = tuple(_parse_tuple_param(hp.get("upsample_rates", (8, 8, 4))))
    upsample_initial_channel = int(hp.get("upsample_initial_channel", 256))
    upsample_kernel_sizes = tuple(_parse_tuple_param(hp.get("upsample_kernel_sizes", (16, 16, 8))))
    gin_channels = int(hp.get("gin_channels", 0))
    if num_speakers > 1 and gin_channels <= 0:
        gin_channels = 512
    use_sdp = bool(hp.get("use_sdp", True))
    use_spectral_norm = bool(hp.get("use_spectral_norm", False))

    model_g = SynthesizerTrn(
        n_vocab=num_symbols,
        spec_channels=filter_length // 2 + 1,
        segment_size=segment_size // hop_length,
        inter_channels=inter_channels,
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
        resblock=resblock,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        n_speakers=num_speakers,
        gin_channels=gin_channels,
        use_sdp=use_sdp,
    )
    model_d = MultiPeriodDiscriminator(use_spectral_norm=use_spectral_norm)

    state_dict = ckpt.get("state_dict", ckpt)
    g_state = {
        k[len("model_g.") :]: v for k, v in state_dict.items() if k.startswith("model_g.")
    }
    d_state = {
        k[len("model_d.") :]: v for k, v in state_dict.items() if k.startswith("model_d.")
    }

    if g_state:
        model_g.load_state_dict(g_state, strict=strict)
    if d_state:
        model_d.load_state_dict(d_state, strict=strict)

    model_g = model_g.to(device)
    model_d = model_d.to(device)
    return model_g, model_d, hp
