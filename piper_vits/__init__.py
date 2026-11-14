"""
Clean, standalone Piper / VITS core package adapted from
MahtaFetrat/Piper-with-LCA-Phonemizer (OHF-Voice/piper1-gpl).
"""

from . import commons
from .checkpoint import build_piper_vits_from_checkpoint
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from .vits_models import (
    DurationPredictor,
    Generator,
    MultiPeriodDiscriminator,
    PosteriorEncoder,
    ResidualCouplingBlock,
    StochasticDurationPredictor,
    SynthesizerTrn,
    TextEncoder,
)

__all__ = [
    "DurationPredictor",
    "Generator",
    "MultiPeriodDiscriminator",
    "PosteriorEncoder",
    "ResidualCouplingBlock",
    "StochasticDurationPredictor",
    "SynthesizerTrn",
    "TextEncoder",
    "build_piper_vits_from_checkpoint",
    "commons",
    "discriminator_loss",
    "feature_loss",
    "generator_loss",
    "kl_loss",
    "mel_spectrogram_torch",
    "spec_to_mel_torch",
    "spectrogram_torch",
]
