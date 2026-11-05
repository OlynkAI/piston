"""Load Piston v0.1 and ask what-if questions of a time-series panel.

    from piston import Piston
    pst = Piston.from_pretrained("OlynkAI/piston-v0.1")      # or a local folder
    out = pst.predict(panel, treatment=0, target=2, lag=1, x0=10.0, x1=12.0)
    # {'effect': 0.41, 'lo90': 0.12, 'hi90': 0.73, 'p_answerable': 0.83, 'p_causal_path': 0.95}

The question: at some time t, set column `treatment` to x1 instead of x0, overriding its usual
drivers. What is the expected change in column `target` at t + lag? Answers are in the target's
own units. Refuse (or ask for more information) when p_answerable < 0.5.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch

from .data import KNOW_MODES, T_MODEL, V_MAX
from .model import CausalModel

MIN_STEPS = 128


class Piston:
    def __init__(self, models: list[CausalModel], device: torch.device):
        self.models, self.device = models, device

    @classmethod
    def from_pretrained(cls, path: str, device: str | None = None) -> "Piston":
        """`path`: a local folder with seed*.pt, or a Hugging Face repo id."""
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, allow_patterns=["weights/*.pt"])
        files = sorted(glob.glob(str(Path(path) / "weights" / "seed*.pt"))) or sorted(glob.glob(str(Path(path) / "seed*.pt")))
        if not files:
            raise FileNotFoundError(f"no seed*.pt checkpoints under {path}")
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        models = []
        for f in files:
            m, ck = CausalModel.load(f, map_location=dev)
            m.to(dev).eval()
            m.calib = ck.get("calib")
            models.append(m)
        return cls(models, dev)

    # ── batched core, shared with the benchmark scorer ───────────────────────

    @torch.no_grad()
    def predict_batch(self, b: dict, know: str = "none") -> np.ndarray:
        """b: arrays as produced by ShardDataset.batch (z-units). Returns (B, 5):
        q05, q50, q95 of the normalised effect, P(answerable), P(causal path)."""
        t = {k: torch.as_tensor(v, device=self.device) for k, v in b.items()}
        P, I, C = [], [], []
        for m in self.models:
            kw = dict(edge=t["edge"], hcc=t["hcc"]) if m.cfg.knowledge else {}
            lg, li, lp = m(t["x"], t["tmask"], t["vmask"], t["treat"], t["target"], t["lag"], t["delta"],
                           t["x0z"], **kw)
            lg = lg.float()
            cal = getattr(m, "calib", None)
            if cal:
                tt = cal["t_eff"][know] if isinstance(cal["t_eff"], dict) else cal["t_eff"]
                lg = lg / tt
            P.append(lg.softmax(-1)); I.append(torch.sigmoid(li.float())); C.append(torch.sigmoid(lp.float()))
        p = torch.stack(P).mean(0)                                   # mixture of the seeds
        q = self.models[0].quantiles(torch.log(p.clamp_min(1e-12)), qs=(0.05, 0.5, 0.95))
        return torch.cat([q, torch.stack(I).mean(0)[:, None], torch.stack(C).mean(0)[:, None]], -1).cpu().numpy()

    # ── one question on a raw panel ──────────────────────────────────────────

    def predict(self, panel, treatment: int, target: int, lag: int, x0: float, x1: float,
                edges=None, hidden=None) -> dict:
        """panel: (steps, variables) array in any units, oldest row first; up to 9 variables; the last
        256 steps are used. lag: 1, 2 or 3 steps. Optional graph knowledge (variables x variables):
          edges[i, j]  0 unknown, 1 "i causes j", 2 "i does not cause j"
          hidden[i, j] 0 unknown, 1 "a hidden cause drives i and j", 2 "no hidden common cause"
        Partial knowledge helps P(answerable). If you know the full graph, run an exact identification
        algorithm yourself and use this model's answer without graph input (best tested setup)."""
        P = np.asarray(panel, dtype=np.float64)
        if P.ndim != 2:
            raise ValueError("panel must be (steps, variables)")
        T, V = P.shape
        if V > V_MAX:
            raise ValueError(f"at most {V_MAX} variables")
        if not 1 <= lag <= 3:
            raise ValueError("lag must be 1, 2 or 3")
        if T < MIN_STEPS:
            raise ValueError(f"need at least {MIN_STEPS} steps (trained on 200-1,000)")
        if not np.isfinite(P).all():
            raise ValueError("panel has missing or infinite values")
        seg = P[-T_MODEL:]
        n = len(seg)
        m, sd = seg.mean(0), seg.std(0) + 1e-6
        x = np.zeros((1, V_MAX, T_MODEL), np.float32)
        x[0, :V, :n] = ((seg - m) / sd).T
        tmask = np.zeros((1, T_MODEL), bool); tmask[0, :n] = True
        vmask = np.zeros((1, V_MAX), bool); vmask[0, :V] = True
        e = np.zeros((1, V_MAX, V_MAX), np.int64)
        h = np.zeros((1, V_MAX, V_MAX), np.int64)
        if edges is not None:
            e[0, :V, :V] = np.asarray(edges, np.int64)
        if hidden is not None:
            hh = np.asarray(hidden, np.int64)
            h[0, :V, :V] = np.maximum(hh, hh.T)
        know = "none" if edges is None and hidden is None else "partial"
        b = dict(x=x, tmask=tmask, vmask=vmask, treat=np.array([treatment]), target=np.array([target]),
                 lag=np.array([lag]), delta=np.array([(x1 - x0) / sd[treatment]], np.float32),
                 x0z=np.array([(x0 - m[treatment]) / sd[treatment]], np.float32), edge=e, hcc=h)
        r = self.predict_batch(b, know=know)[0]
        s = float(sd[target])
        return {"effect": float(r[1]) * s, "lo90": float(r[0]) * s, "hi90": float(r[2]) * s,
                "p_answerable": float(r[3]), "p_causal_path": float(r[4])}


__all__ = ["Piston", "KNOW_MODES"]
