"""Episode -> fixed-size arrays for the model, and the training dataset.

Normalisation contract (unit and scale free, R2):
  every variable is z-scored on the data the model sees; the question and the
  answer are expressed in those z-units:
    delta = (x1 - x0) / sd_X,   x0z = (x0 - mean_X) / sd_X,   tau_n = tau / sd_Y
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

V_MAX = 9          # prior max observed variables
T_KEEP = 384       # steps stored per training episode
T_MODEL = 256      # steps the model reads
K_MAX = 3          # max effect lag
KNOW_MODES = ("none", "partial", "full")
# Knowledge states the model can be given about each ordered pair of variables (i, j):
#   edge  0 unknown, 1 "i causes j" (at some lag; i == j means its own past), 2 "i does not cause j"
#   hcc   0 unknown, 1 "a hidden cause drives both", 2 "no hidden common cause"


def episode_to_arrays(ep: dict, rng: np.random.Generator | None, t_keep: int = T_KEEP):
    """Crop (random if rng, else the last t_keep steps), z-score, convert queries.
    Returns dict of numpy arrays, or None if the episode has no usable query."""
    P = np.asarray(ep["panel"], dtype=np.float64)            # (T, V) panel units
    T, V = P.shape
    if T > t_keep:
        s = int(rng.integers(0, T - t_keep + 1)) if rng is not None else T - t_keep
        P = P[s:s + t_keep]
    mu, sd = P.mean(0), P.std(0) + 1e-12
    Z = (P - mu) / sd
    names = {n: i for i, n in enumerate(ep["variables"])}
    rows = []
    for q in ep["queries"]:
        X, Y = names[q["treatment"]], names[q["target"]]
        rows.append((X, Y, q["lag"], (q["x1"] - q["x0"]) / sd[X], (q["x0"] - mu[X]) / sd[X],
                     q["tau"] / sd[Y], q["tau_se"] / sd[Y],
                     1.0 if q["label"] == "identified" else 0.0, 1.0 if q["causal_path"] else 0.0))
    if not rows:
        return None
    panel = np.zeros((t_keep, V_MAX), np.float32)
    panel[:Z.shape[0], :V] = Z
    adj = np.zeros((V_MAX, V_MAX), np.uint8)
    hcc = np.zeros((V_MAX, V_MAX), np.uint8)
    g = ep.get("graph") or {}
    for a, b, _lag in g.get("directed", []):
        if a in names and b in names:
            adj[names[a], names[b]] = 1
    for a, b, _gap in g.get("hidden_common_causes", []):
        if a in names and b in names:
            hcc[names[a], names[b]] = hcc[names[b], names[a]] = 1
    q = np.array(rows, dtype=np.float64)
    return dict(panel=panel, t_len=Z.shape[0], n_vars=V, adj=adj, hcc=hcc, treat=q[:, 0].astype(np.int8),
                target=q[:, 1].astype(np.int8), lag=q[:, 2].astype(np.int8),
                delta=q[:, 3].astype(np.float32), x0z=q[:, 4].astype(np.float32),
                tau=q[:, 5].astype(np.float32), tau_se=q[:, 6].astype(np.float32),
                ident=q[:, 7].astype(np.float32), path=q[:, 8].astype(np.float32),
                family=ep["family_id"], episode_id=ep["episode_id"])


def write_shard(items: list[dict], path: Path) -> None:
    qi = np.concatenate([np.full(len(it["treat"]), i, np.int32) for i, it in enumerate(items)])
    cat = lambda k: np.concatenate([it[k] for it in items])
    np.savez_compressed(
        path, panel=np.stack([it["panel"] for it in items]).astype(np.float16),
        t_len=np.array([it["t_len"] for it in items], np.int16),
        n_vars=np.array([it["n_vars"] for it in items], np.int8),
        adj=np.stack([it["adj"] for it in items]), hcc=np.stack([it["hcc"] for it in items]),
        q_ep=qi, treat=cat("treat"), target=cat("target"), lag=cat("lag"), delta=cat("delta"),
        x0z=cat("x0z"), tau=cat("tau"), tau_se=cat("tau_se"), ident=cat("ident"), path=cat("path"),
        family=np.array([it["family"] for it in items]), episode_id=np.array([it["episode_id"] for it in items]))


def load_jsonl_dir(d: str | Path, rng=None, limit=None) -> list[dict]:
    out = []
    for p in sorted(Path(d).glob("shard_*.jsonl.gz")):
        with gzip.open(p, "rt") as f:
            for line in f:
                a = episode_to_arrays(json.loads(line), rng)
                if a is not None:
                    out.append(a)
                if limit and len(out) >= limit:
                    return out
    return out


class ShardDataset:
    """Loads npz shards into memory; yields query-level samples with a fresh
    random T_MODEL crop each time (re-standardised on the crop)."""

    def __init__(self, paths: list[str | Path]):
        parts = [np.load(p) for p in paths]
        self.panel = np.concatenate([p["panel"] for p in parts])
        self.t_len = np.concatenate([p["t_len"] for p in parts]).astype(np.int64)
        self.n_vars = np.concatenate([p["n_vars"] for p in parts]).astype(np.int64)
        off, qe = 0, []
        for p in parts:
            qe.append(p["q_ep"].astype(np.int64) + off)
            off += len(p["panel"])
        self.q_ep = np.concatenate(qe)
        for k in ("treat", "target", "lag", "delta", "x0z", "tau", "tau_se", "ident", "path"):
            setattr(self, k, np.concatenate([p[k] for p in parts]))
        self.family = np.concatenate([p["family"] for p in parts])
        self.has_graph = all("adj" in p for p in parts)
        if self.has_graph:
            self.adj = np.concatenate([p["adj"] for p in parts])
            self.hcc = np.concatenate([p["hcc"] for p in parts])

    def __len__(self):
        return len(self.q_ep)

    def batch(self, idx: np.ndarray, rng: np.random.Generator | None, t_model: int = T_MODEL,
              know: str = "none", know_rng: np.random.Generator | None = None):
        """know: 'none' | 'partial' | 'full' | 'mix' (training: a third each). Partial reveals each
        ordered pair with probability U(0.2, 0.8). Datasets without graphs always give 'none'."""
        B = len(idx)
        x = np.zeros((B, V_MAX, t_model), np.float32)
        tmask = np.zeros((B, t_model), bool)
        vmask = np.zeros((B, V_MAX), bool)
        delta = np.zeros(B, np.float32)
        x0z = np.zeros(B, np.float32)
        tau = np.zeros(B, np.float32)
        edge = np.zeros((B, V_MAX, V_MAX), np.int64)
        hcc = np.zeros((B, V_MAX, V_MAX), np.int64)
        krng = know_rng if know_rng is not None else (rng if rng is not None else np.random.default_rng(0))
        for b, i in enumerate(idx):
            e = self.q_ep[i]
            L, V = int(self.t_len[e]), int(self.n_vars[e])
            n = min(L, t_model)
            s = int(rng.integers(0, L - n + 1)) if rng is not None else L - n
            seg = self.panel[e, s:s + n, :V].astype(np.float32)
            m, sd = seg.mean(0), seg.std(0) + 1e-6
            x[b, :V, :n] = ((seg - m) / sd).T
            tmask[b, :n] = True
            vmask[b, :V] = True
            X, Y = int(self.treat[i]), int(self.target[i])
            delta[b] = self.delta[i] / sd[X]
            x0z[b] = (self.x0z[i] - m[X]) / sd[X]
            tau[b] = self.tau[i] / sd[Y]
            mode = know if know != "mix" else KNOW_MODES[int(krng.integers(0, 3))]
            if self.has_graph and mode != "none":
                if mode == "full":
                    reveal = np.ones((V, V), bool)
                else:
                    reveal = krng.random((V, V)) < krng.uniform(0.2, 0.8)
                ea = self.adj[e, :V, :V].astype(np.int64)
                ha = self.hcc[e, :V, :V].astype(np.int64)
                edge[b, :V, :V] = np.where(reveal, np.where(ea == 1, 1, 2), 0)
                hr = reveal | reveal.T
                hcc[b, :V, :V] = np.where(hr, np.where(ha == 1, 1, 2), 0)
        return dict(x=x, tmask=tmask, vmask=vmask, treat=self.treat[idx].astype(np.int64),
                    target=self.target[idx].astype(np.int64), lag=self.lag[idx].astype(np.int64),
                    delta=delta, x0z=x0z, tau=tau, ident=self.ident[idx], path=self.path[idx],
                    edge=edge, hcc=hcc)
