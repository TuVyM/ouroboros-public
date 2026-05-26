from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(5, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            norm_first=True,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        # x: (B, T, 5)  T == seq_len (or shorter with padding mask)
        x = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        if return_sequence:
            return x  # (B, T, d_model)
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return x.mean(dim=1)  # (B, d_model)


class MultiScaleEncoder(nn.Module):
    SEQ_1H = 168   # 1h bars
    SEQ_4H = 42    # 4h bars  (168 / 4)
    SEQ_1D = 30    # 1d bars  (720 / 24)
    WIN_1D = 720   # 1h bars needed to produce SEQ_1D daily bars

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.enc_1h = TransformerEncoder(self.SEQ_1H, d_model)
        self.enc_4h = TransformerEncoder(self.SEQ_4H, d_model)
        self.enc_1d = TransformerEncoder(self.SEQ_1D, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _znorm(x: torch.Tensor) -> torch.Tensor:
        # Per-window, per-channel z-score normalisation.  x: (B, T, 5)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std

    @staticmethod
    def _resample(x: torch.Tensor, factor: int) -> torch.Tensor:
        # Aggregate T 1h bars into T//factor higher-timeframe bars.  x: (B, T, 5)
        B, T, _ = x.shape
        n = T // factor
        r = x[:, : n * factor, :].reshape(B, n, factor, 5)
        o = r[:, :, 0, 0:1]
        h = r[:, :, :, 1].max(dim=2).values.unsqueeze(-1)
        l = r[:, :, :, 2].min(dim=2).values.unsqueeze(-1)
        c = r[:, :, -1, 3:4]
        v = r[:, :, :, 4].sum(dim=2).unsqueeze(-1)
        return torch.cat([o, h, l, c, v], dim=-1)  # (B, n, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 720, 5) — zero-padded when fewer than 720 bars available
        x1h = x[:, -self.SEQ_1H :, :]                              # (B, 168, 5)
        x4h = self._resample(x[:, -(self.SEQ_4H * 4) :, :], 4)    # (B, 42, 5)
        x1d = self._resample(x, 24)[:, -self.SEQ_1D :, :]          # (B, 30, 5)
        e1 = self.enc_1h(self._znorm(x1h))
        e4 = self.enc_4h(self._znorm(x4h))
        ed = self.enc_1d(self._znorm(x1d))
        return self.fusion(torch.cat([e1, e4, ed], dim=-1))         # (B, 64)

    def batch_embed(
        self,
        ohlcv: np.ndarray,
        out_indices: np.ndarray,
        batch_size: int = 512,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Compute embeddings for multiple output rows in one batched pass.

        out_indices: 1-D int array of ohlcv row indices.
            For build_features output row i, pass WINDOW + i.
        Returns: (len(out_indices), 64) float32
        """
        self.eval()
        parts: list[np.ndarray] = []
        for start in range(0, len(out_indices), batch_size):
            batch_idx = out_indices[start : start + batch_size]
            B = len(batch_idx)
            windows = np.zeros((B, self.WIN_1D, 5), dtype=np.float32)
            for j, idx in enumerate(batch_idx):
                lo = max(0, idx - (self.WIN_1D - 1))
                win = ohlcv[lo : idx + 1].astype(np.float32)
                windows[j, self.WIN_1D - len(win) :] = win
            xt = torch.from_numpy(windows).to(device)
            with torch.no_grad():
                out = self(xt)
            parts.append(out.cpu().numpy())
        return np.concatenate(parts, axis=0).astype(np.float32)

    @torch.no_grad()
    def embed(self, htf_1h: np.ndarray) -> np.ndarray:
        """Single-sample inference. htf_1h: (N, 5) float32, N should be >= 720."""
        self.eval()
        N = len(htf_1h)
        arr = htf_1h[-self.WIN_1D :].astype(np.float32) if N >= self.WIN_1D else htf_1h.astype(np.float32)
        if len(arr) < self.WIN_1D:
            pad = np.zeros((self.WIN_1D - len(arr), 5), dtype=np.float32)
            arr = np.concatenate([pad, arr], axis=0)
        x = torch.from_numpy(arr).unsqueeze(0)  # (1, 720, 5)
        out = self(x)
        return out.squeeze(0).numpy().astype(np.float32)
