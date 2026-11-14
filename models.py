"""
Neural Architecture Upgrades for Persian Piper / VITS Fine-Tuning.

All 3 custom upgrades wrap pre-trained Piper VITS (`SynthesizerTrn`) modules using
strict zero-initialization so Step-0 preserves 100% of the base Persian checkpoint
weights (`0.000000` Step-0 diff) while enabling:
  1. ProsodyEnhancedEmbedding: Zero-init 5-tag Persian Prosody Embedding + LayerNorm
     512-dim Homo-GE2PE ByT5 Context Projection (masked off blank separator tokens).
  2. DurationPredictorAdapter: Zero-init ConvNeXt + GLU Residual Adapter wrapping both
     StochasticDurationPredictor (`reverse=False` training loss / `reverse=True` ONNX)
     and deterministic DurationPredictor variants.
  3. InvertibleFlowAttentionCoupling (+ FlowWithAttention): Bijective Multi-Head
     Self-Attention Normalizing Flow coupling (ONNX-compatible pure QKV attention).
"""

import copy
import inspect
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from symbols import NUM_PROSODY_TAGS


class ProsodyEnhancedEmbedding(nn.Module):
    """
    Upgrade #1: Zero-Initialized Persian Prosody + Contextual G2P Embedding.

    Wraps pre-trained `enc_p.emb` (`256 x 192`) and adds:
      - Zero-initialized `prosody_emb` (`5 x 192`: Normal, Stress, Tashdid, Ezafe, Question)
      - LayerNorm + Zero-initialized `context_proj` projecting 1472-dim (ByT5-small)
        `Homo-GE2PE` encoder states to 192-dim acoustic prior deltas.
    Masked by `(x != 0)` so VITS blank separator tokens (`_`, id=0) stay untouched.
    """

    def __init__(
        self,
        pretrained_phoneme_emb: nn.Embedding,
        num_prosody_tags: int = NUM_PROSODY_TAGS,
        context_dim: int = 1472,
    ):
        super().__init__()
        self.phoneme_emb = pretrained_phoneme_emb
        self.embedding_dim = pretrained_phoneme_emb.embedding_dim
        self.num_embeddings = pretrained_phoneme_emb.num_embeddings
        self.context_dim = context_dim

        self.prosody_emb = nn.Embedding(num_prosody_tags, self.embedding_dim)
        nn.init.zeros_(self.prosody_emb.weight)

        self.context_norm = nn.LayerNorm(context_dim)
        self.context_proj = nn.Linear(context_dim, self.embedding_dim)
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

        self._pending_prosody_ids: Optional[torch.Tensor] = None
        self._pending_context_emb: Optional[torch.Tensor] = None

    def set_prosody_ids(self, prosody_ids: Optional[torch.Tensor]) -> None:
        self._pending_prosody_ids = prosody_ids

    def set_context_embedding(self, context_emb: Optional[torch.Tensor]) -> None:
        self._pending_context_emb = context_emb

    def set_conditioning(
        self,
        prosody_ids: Optional[torch.Tensor] = None,
        context_emb: Optional[torch.Tensor] = None,
    ) -> None:
        self._pending_prosody_ids = prosody_ids
        self._pending_context_emb = context_emb

    def clear_conditioning(self) -> None:
        self._pending_prosody_ids = None
        self._pending_context_emb = None

    def forward(
        self,
        x: torch.Tensor,
        prosody_ids: Optional[torch.Tensor] = None,
        context_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base = self.phoneme_emb(x)  # [B, T, emb_dim]

        p_ids = prosody_ids if prosody_ids is not None else self._pending_prosody_ids
        ctx = context_emb if context_emb is not None else self._pending_context_emb
        non_pad = (x != 0).unsqueeze(-1).type_as(base)

        if p_ids is not None:
            p_ids = p_ids.to(device=x.device, dtype=torch.long)
            if p_ids.shape == x.shape:
                base = base + self.prosody_emb(p_ids) * non_pad

        if ctx is not None:
            if ctx.dim() != 3 or ctx.size(-1) != self.context_dim:
                raise ValueError(
                    f"Expected 3D context_emb with last dim {self.context_dim}, got {tuple(ctx.shape)}"
                )
            ctx = ctx.to(device=base.device, dtype=base.dtype)
            if ctx.size(0) == 1 and x.size(0) > 1:
                ctx = ctx.expand(x.size(0), -1, -1)
            if not torch.onnx.is_in_onnx_export() and ctx.size(1) != x.size(1):
                ctx = F.interpolate(
                    ctx.transpose(1, 2),
                    size=x.size(1),
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            base = base + self.context_proj(self.context_norm(ctx)) * non_pad

        return base


class ConvNeXtDurationBlock(nn.Module):
    """Lightweight Depthwise Conv1D + LayerNorm + GLU residual block."""

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.pw_in = nn.Linear(channels, channels * 2)
        self.pw_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.dwconv(x * x_mask)
        h = self.norm(h.transpose(1, 2))
        a, b = self.pw_in(h).chunk(2, dim=-1)
        h = a * torch.sigmoid(b)
        h = self.pw_out(h).transpose(1, 2)
        return (residual + h) * x_mask


class DurationPredictorAdapter(nn.Module):
    """
    Upgrade #2: ConvNeXt + GLU Residual Duration Adapter wrapping pre-trained DP / SDP.

    When wrapping Piper's `StochasticDurationPredictor`:
      - Training (`reverse=False`, `w` provided): adds MSE supervision on the residual
        `delta_logw` against `(log(w) - deterministic_base_logw)` to `base_nll` (`[B]`).
      - Inference (`reverse=True`): returns `(base_logw + delta_logw) * x_mask` (`[B, 1, T]`).
    """

    def __init__(
        self,
        pretrained_dp: nn.Module,
        in_channels: int = 192,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.pretrained_dp = pretrained_dp
        self.in_channels = in_channels
        self.gin_channels = gin_channels
        self._is_sdp = "reverse" in inspect.signature(pretrained_dp.forward).parameters

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)
        self.adapter_blocks = nn.ModuleList(
            [ConvNeXtDurationBlock(in_channels, kernel_size=5) for _ in range(2)]
        )
        self.out_proj = nn.Conv1d(in_channels, 1, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _compute_delta(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x * x_mask
        if g is not None and self.gin_channels > 0:
            h = h + self.cond(g.detach())
        for blk in self.adapter_blocks:
            h = blk(h, x_mask)
        return self.out_proj(h) * x_mask

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        if self._is_sdp:
            if not reverse:
                base_nll = self.pretrained_dp(
                    x, x_mask, w=w, g=g, reverse=False, noise_scale=noise_scale
                )
                delta_logw = self._compute_delta(x, x_mask, g=g)
                if w is not None:
                    with torch.no_grad():
                        base_logw_mode = self.pretrained_dp(
                            x, x_mask, g=g, reverse=True, noise_scale=0.0
                        )
                        target_delta = (
                            torch.log(torch.clamp(w, min=1e-6)) - base_logw_mode
                        ) * x_mask
                    adapter_loss = torch.sum(((delta_logw - target_delta) ** 2) * x_mask, [1, 2])
                else:
                    adapter_loss = delta_logw.sum() * 0.0
                return base_nll + adapter_loss

            base_logw = self.pretrained_dp(
                x, x_mask, g=g, reverse=True, noise_scale=noise_scale
            )
            delta_logw = self._compute_delta(x, x_mask, g=g)
            return (base_logw + delta_logw) * x_mask

        try:
            base_logw = self.pretrained_dp(x, x_mask, g=g)
        except TypeError:
            base_logw = self.pretrained_dp(x * x_mask)
        delta_logw = self._compute_delta(x, x_mask, g=g)
        return (base_logw + delta_logw) * x_mask


class InvertibleFlowAttentionCoupling(nn.Module):
    """
    Upgrade #3: Strictly Bijective Normalizing Flow Multi-Head Self-Attention Coupling.

    Uses pure matrix QKV scaled-dot-product attention for exact float32 invertibility
    and reliable opset-15 ONNX tracing across dynamic sequence lengths.
    """

    def __init__(self, channels: int = 192, n_heads: int = 4, dropout: float = 0.05):
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2
        self.n_heads = n_heads
        assert self.half_channels % n_heads == 0, "half_channels must be divisible by n_heads"
        self.head_dim = self.half_channels // n_heads

        self.norm = nn.LayerNorm(self.half_channels)
        self.qkv_proj = nn.Linear(self.half_channels, self.half_channels * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.half_channels, self.half_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _compute_shift(self, x0: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        b, c, t = x0.shape
        x0_t = self.norm(x0.transpose(1, 2))  # [B, T, C/2]
        q, k, v = self.qkv_proj(x0_t).chunk(3, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_mask = x_mask.unsqueeze(1)  # [B, 1, 1, T]
        scores = scores.masked_fill(key_mask == 0, -1e4)
        attn = self.attn_drop(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, c)
        delta = self.out_proj(out).transpose(1, 2)
        return delta * x_mask

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        x0, x1 = x.chunk(2, dim=1)
        shift = self._compute_shift(x0, x_mask)
        if not reverse:
            y1 = (x1 + shift) * x_mask
        else:
            y1 = (x1 - shift) * x_mask
        return torch.cat([x0, y1], dim=1)


class FlowWithAttention(nn.Module):
    """Wraps Piper's `ResidualCouplingBlock` in exact forward/inverse composition order."""

    def __init__(self, base_flow: nn.Module, attn_coupling: InvertibleFlowAttentionCoupling):
        super().__init__()
        self.base_flow = base_flow
        self.attn_coupling = attn_coupling

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            x = self.base_flow(x, x_mask, g=g, reverse=False)
            x = self.attn_coupling(x, x_mask, g=g, reverse=False)
            return x
        x = self.attn_coupling(x, x_mask, g=g, reverse=True)
        x = self.base_flow(x, x_mask, g=g, reverse=True)
        return x


class UpgradedPersianVits(nn.Module):
    """
    Side-effect-free wrapper around upgraded Piper `SynthesizerTrn` supporting
    batched GPU fine-tuning with `(prosody_ids, context_emb)` and clean ONNX export.
    """

    def __init__(self, vits_model: nn.Module, context_dim: int = 1472):
        super().__init__()
        attach_upgrades_to_piper_vits(vits_model, context_dim=context_dim)
        self.vits = vits_model

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        prosody_ids: Optional[torch.Tensor] = None,
        context_emb: Optional[torch.Tensor] = None,
        sid: Optional[torch.Tensor] = None,
    ):
        self.vits.enc_p.emb.set_conditioning(prosody_ids, context_emb)
        try:
            return self.vits(x, x_lengths, y, y_lengths, sid=sid)
        finally:
            self.vits.enc_p.emb.clear_conditioning()

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        prosody_ids: Optional[torch.Tensor] = None,
        context_emb: Optional[torch.Tensor] = None,
        sid: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        max_len: Optional[int] = None,
    ):
        self.vits.enc_p.emb.set_conditioning(prosody_ids, context_emb)
        try:
            return self.vits.infer(
                x,
                x_lengths,
                sid=sid,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w,
                max_len=max_len,
            )
        finally:
            self.vits.enc_p.emb.clear_conditioning()


class _ONNXExportWrapper(nn.Module):
    """Pure functional ONNX tracing wrapper (no mutable state or weight_norm hooks)."""

    def __init__(self, upgraded_vits: UpgradedPersianVits):
        super().__init__()
        src = upgraded_vits.vits
        ctx_dim = src.enc_p.emb.context_dim
        fresh = type(src)(
            n_vocab=src.n_vocab,
            spec_channels=src.spec_channels,
            segment_size=src.segment_size,
            inter_channels=src.inter_channels,
            hidden_channels=src.hidden_channels,
            filter_channels=src.filter_channels,
            n_heads=src.n_heads,
            n_layers=src.n_layers,
            kernel_size=src.kernel_size,
            p_dropout=src.p_dropout,
            resblock=src.resblock,
            resblock_kernel_sizes=src.resblock_kernel_sizes,
            resblock_dilation_sizes=src.resblock_dilation_sizes,
            upsample_rates=src.upsample_rates,
            upsample_initial_channel=src.upsample_initial_channel,
            upsample_kernel_sizes=src.upsample_kernel_sizes,
            n_speakers=src.n_speakers,
            gin_channels=src.gin_channels,
            use_sdp=src.use_sdp,
        )
        attach_upgrades_to_piper_vits(fresh, context_dim=ctx_dim)
        fresh.load_state_dict(
            {k: v.detach().cpu() for k, v in src.state_dict().items()}, strict=True
        )
        fresh.eval()
        with torch.no_grad():
            for mod in fresh.modules():
                try:
                    nn.utils.remove_weight_norm(mod)
                except ValueError:
                    pass
        fresh.enc_p.emb.clear_conditioning()
        self.vits = fresh

    def _encode_prior(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        prosody_ids: torch.Tensor,
        context_emb: torch.Tensor,
    ):
        from piper_vits import commons

        enc_p = self.vits.enc_p
        h = enc_p.emb(x, prosody_ids=prosody_ids, context_emb=context_emb) * math.sqrt(
            enc_p.hidden_channels
        )
        h = torch.transpose(h, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, h.size(2)), 1).type_as(h)
        h = enc_p.encoder(h * x_mask, x_mask)
        stats = enc_p.proj(h) * x_mask
        m_p, logs_p = torch.split(stats, enc_p.out_channels, dim=1)
        return h, m_p, logs_p, x_mask

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        prosody_ids: torch.Tensor,
        context_emb: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        from piper_vits import commons

        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]

        x, m_p, logs_p, x_mask = self._encode_prior(text, text_lengths, prosody_ids, context_emb)
        g = None
        if getattr(self.vits, "use_sdp", True):
            logw = self.vits.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.vits.dp(x, x_mask, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.vits.flow(z_p, y_mask, g=g, reverse=True)
        audio = self.vits.dec(z * y_mask, g=g).unsqueeze(1)
        return audio


def align_context_embedding(
    context_emb: Optional[torch.Tensor],
    target_len: int,
    context_dim: int = 1472,
) -> torch.Tensor:
    """Align `[B, L_byt5, D]` Homo-GE2PE states to `[B, target_len, D]` for ONNX/VITS."""
    if context_emb is None:
        return torch.zeros((1, target_len, context_dim), dtype=torch.float32)
    ctx = context_emb.detach().float()
    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0)
    if ctx.size(1) != target_len:
        ctx = F.interpolate(
            ctx.transpose(1, 2),
            size=int(target_len),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return ctx


def export_upgraded_vits_onnx(
    upgraded_model: UpgradedPersianVits,
    output_path: str,
    opset_version: int = 15,
) -> str:
    """Export fine-tuned UpgradedPersianVits (including all 3 neural upgrades) to ONNX."""
    export_mod = _ONNXExportWrapper(upgraded_model)
    export_mod.eval()

    vocab_size = getattr(export_mod.vits.enc_p.emb, "num_embeddings", 256)
    ctx_dim = export_mod.vits.enc_p.emb.context_dim
    dummy_t = 32
    dummy_input = (
        torch.randint(low=0, high=vocab_size, size=(1, dummy_t), dtype=torch.long),
        torch.LongTensor([dummy_t]),
        torch.zeros((1, dummy_t), dtype=torch.long),
        torch.zeros((1, dummy_t, ctx_dim), dtype=torch.float32),
        torch.FloatTensor([0.667, 1.0, 0.8]),
    )

    export_kwargs = dict(
        f=output_path,
        verbose=False,
        opset_version=opset_version,
        input_names=["input", "input_lengths", "prosody_ids", "context_emb", "scales"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "prosody_ids": {0: "batch_size", 1: "phonemes"},
            "context_emb": {0: "batch_size", 1: "phonemes"},
            "output": {0: "batch_size", 3: "time"},
        },
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(export_mod, dummy_input, **export_kwargs)
    return output_path


def attach_upgrades_to_piper_vits(vits_model: nn.Module, context_dim: int = 1472) -> nn.Module:
    """
    Safely wraps a pre-trained Piper / VITS generator with all 3 Persian upgrades
    while preserving 100% of pre-trained weights at Epoch 0.
    """
    enc = getattr(vits_model, "enc_p", None) or getattr(vits_model, "text_encoder", None)
    hidden_ch = getattr(enc, "hidden_channels", 192)
    inter_ch = getattr(vits_model, "inter_channels", hidden_ch)
    gin_ch = getattr(vits_model, "gin_channels", 0)

    # 1. Upgrade #1: Zero-init Prosody + 512-dim Context Embedding
    if enc is not None and hasattr(enc, "emb") and not isinstance(enc.emb, ProsodyEnhancedEmbedding):
        enc.emb = ProsodyEnhancedEmbedding(enc.emb, context_dim=context_dim)

    # 2. Upgrade #2: Wrap pre-trained DurationPredictor / StochasticDurationPredictor
    if hasattr(vits_model, "dp") and not isinstance(vits_model.dp, DurationPredictorAdapter):
        vits_model.dp = DurationPredictorAdapter(
            vits_model.dp, in_channels=hidden_ch, gin_channels=gin_ch
        )
    elif hasattr(vits_model, "duration_predictor") and not isinstance(
        vits_model.duration_predictor, DurationPredictorAdapter
    ):
        vits_model.duration_predictor = DurationPredictorAdapter(
            vits_model.duration_predictor, in_channels=hidden_ch, gin_channels=gin_ch
        )

    # 3. Upgrade #3: Wrap Normalizing Flow with Bijective Multi-Head Self-Attention
    attn_coupling = InvertibleFlowAttentionCoupling(channels=inter_ch, n_heads=4)
    if hasattr(vits_model, "flow") and not isinstance(vits_model.flow, FlowWithAttention):
        vits_model.flow = FlowWithAttention(vits_model.flow, attn_coupling)
    elif not hasattr(vits_model, "flow"):
        vits_model.flow_attention_coupling = attn_coupling

    return vits_model
