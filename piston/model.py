"""Piston v0.1: numeric in-context causal model.

Reads a z-scored panel (variables x time) and one what-if question, returns:
  - a distribution over the normalised effect tau / sd_Y (bar distribution),
  - P(identified): can this panel answer the question at all,
  - P(causal path): does the treatment reach the target at that lag.

Axial transformer: attention across time patches within a variable, then
across variables within a time patch. No variable position embedding, so the
output does not depend on variable order (R1). Roles (treatment / target)
are embeddings added to that variable's tokens.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d: int = 192
    layers: int = 6
    heads: int = 6
    patch: int = 16
    t_model: int = 256
    v_max: int = 9
    k_max: int = 3
    n_bins: int = 128
    tau_lim: float = 8.0
    bin_curve: float = 3.0
    dropout: float = 0.0
    knowledge: bool = False   # v2: accept what the user knows about the graph


def bin_edges(cfg: Config) -> torch.Tensor:
    """Symmetric sinh-spaced edges: fine near 0, coarse in the tails. 0 is an edge."""
    u = torch.linspace(-1, 1, cfg.n_bins + 1, dtype=torch.float64)
    e = cfg.tau_lim * torch.sinh(cfg.bin_curve * u) / math.sinh(cfg.bin_curve)
    return e.float()


class AxialBlock(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.ln_t, self.ln_v, self.ln_m = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.att_t = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.att_v = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))

    def forward(self, h, tpad, vpad, vbias=None):
        """vbias (B, heads, V, V) additive attention bias from graph knowledge, or None."""
        B, V, P, d = h.shape
        x = self.ln_t(h).reshape(B * V, P, d)
        kpm = tpad[:, None, :].expand(B, V, P).reshape(B * V, P)
        a = self.att_t(x, x, x, key_padding_mask=kpm, need_weights=False)[0]
        h = h + a.reshape(B, V, P, d)
        x = self.ln_v(h).permute(0, 2, 1, 3).reshape(B * P, V, d)
        if vbias is None:
            kpm = vpad[:, None, :].expand(B, P, V).reshape(B * P, V)
            a = self.att_v(x, x, x, key_padding_mask=kpm, need_weights=False)[0]
        else:
            H = vbias.shape[1]
            msk = vbias.float().masked_fill(vpad[:, None, None, :], float("-inf"))
            msk = msk[:, None].expand(B, P, H, V, V).reshape(B * P * H, V, V).to(x.dtype)
            a = self.att_v(x, x, x, attn_mask=msk, need_weights=False)[0]
        h = h + a.reshape(B, P, V, d).permute(0, 2, 1, 3)
        return h + self.mlp(self.ln_m(h))


class CausalModel(nn.Module):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg = cfg or Config()
        d, n_p = cfg.d, cfg.t_model // cfg.patch
        self.patch_in = nn.Linear(2 * cfg.patch, d)
        self.pos_t = nn.Parameter(torch.randn(n_p, d) * 0.02)
        self.role = nn.Embedding(4, d)            # 0 other, 1 treatment, 2 target, 3 both
        n_know = 9 if cfg.knowledge else 0      # one-hots: edge X->Y, edge Y->X, hidden cause X~Y
        self.q_in = nn.Sequential(nn.Linear(2 + cfg.k_max + 1 + n_know, d), nn.GELU(), nn.Linear(d, d))
        if cfg.knowledge:
            # per layer, per head: bias for variable i attending to j, from (edge i->j, edge j->i, hidden i~j)
            self.know_bias = nn.Embedding(27, cfg.layers * cfg.heads)
            nn.init.zeros_(self.know_bias.weight)
        self.blocks = nn.ModuleList([AxialBlock(d, cfg.heads, cfg.dropout) for _ in range(cfg.layers)])
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.GELU())
        self.effect = nn.Linear(d, cfg.n_bins)
        self.ident = nn.Linear(d, 1)
        self.path = nn.Linear(d, 1)
        self.register_buffer("edges", bin_edges(cfg), persistent=False)

    def forward(self, x, tmask, vmask, treat, target, lag, delta, x0z, edge=None, hcc=None):
        """x (B,V,T) z-scored; tmask (B,T) valid steps; vmask (B,V) valid vars.
        edge, hcc (B,V,V) knowledge states (0 unknown, 1 yes, 2 no); used when cfg.knowledge."""
        cfg = self.cfg
        B, V, T = x.shape
        P = T // cfg.patch
        m = tmask[:, None, :].expand(B, V, T).float() * vmask[:, :, None].float()
        xp = torch.cat([(x * m).reshape(B, V, P, cfg.patch), m.reshape(B, V, P, cfg.patch)], -1)
        h = self.patch_in(xp) + self.pos_t[None, None, :P]
        ar = torch.arange(V, device=x.device)[None]
        role = (ar == treat[:, None]).long() + 2 * (ar == target[:, None]).long()
        h = h + self.role(role)[:, :, None, :]
        q = [delta[:, None], x0z[:, None].clamp(-6, 6),
             F.one_hot(lag.clamp(0, cfg.k_max), cfg.k_max + 1).float()]
        vbias = None
        if cfg.knowledge:
            if edge is None:
                edge = torch.zeros(B, V, V, dtype=torch.long, device=x.device)
                hcc = torch.zeros_like(edge)
            bi0 = torch.arange(B, device=x.device)
            q += [F.one_hot(edge[bi0, treat, target], 3).float(), F.one_hot(edge[bi0, target, treat], 3).float(),
                  F.one_hot(hcc[bi0, treat, target], 3).float()]
            combo = edge * 9 + edge.transpose(1, 2) * 3 + hcc                 # (B,V,V) in 0..26
            vb = self.know_bias(combo).reshape(B, V, V, cfg.layers, cfg.heads)
            vbias = vb.permute(3, 0, 4, 1, 2)                                  # (L,B,H,V,V)
        qe = self.q_in(torch.cat(q, -1))
        h = h + qe[:, None, None, :]
        tpad = ~(tmask.reshape(B, P, cfg.patch).any(-1))
        vpad = ~vmask
        for li, blk in enumerate(self.blocks):
            h = blk(h, tpad, vpad, None if vbias is None else vbias[li])
        h = self.ln_out(h)
        pm = (~tpad).float()[:, None, :, None]
        per_var = (h * pm).sum(2) / pm.sum(2).clamp(min=1)            # (B,V,d)
        bi = torch.arange(B, device=x.device)
        vm = vmask.float()[:, :, None]
        glob = (per_var * vm).sum(1) / vm.sum(1)
        z = self.head(torch.cat([per_var[bi, treat], per_var[bi, target], glob, qe], -1))
        return self.effect(z), self.ident(z).squeeze(-1), self.path(z).squeeze(-1)

    # ── distribution utilities ──────────────────────────────────────────────

    def target_bins(self, tau):
        t = tau.clamp(self.edges[0] + 1e-4, self.edges[-1] - 1e-4)
        return (torch.bucketize(t, self.edges, right=True) - 1).clamp(0, self.cfg.n_bins - 1)

    def quantiles(self, logits, qs=(0.05, 0.5, 0.95)):
        p = logits.float().softmax(-1)
        cdf = torch.cat([torch.zeros_like(p[:, :1]), p.cumsum(-1)], -1)     # at edges
        out = []
        for qv in qs:
            j = (torch.searchsorted(cdf, torch.full_like(cdf[:, :1], qv)) - 1).clamp(0, self.cfg.n_bins - 1)
            lo, hi = cdf.gather(1, j), cdf.gather(1, j + 1)
            frac = ((qv - lo) / (hi - lo).clamp(min=1e-9)).clamp(0, 1)
            el, eh = self.edges[j.squeeze(1)], self.edges[j.squeeze(1) + 1]
            out.append(el + frac.squeeze(1) * (eh - el))
        return torch.stack(out, -1)

    def save(self, path, extra=None):
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(Config(**ck["cfg"]))
        m.load_state_dict(ck["state"])
        return m, ck
