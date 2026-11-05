"""Trust and effect metrics used by the Piston benchmark, computed on normalised effects.

Per question we need: tau (truth), q05/q50/q95 (model), p_id (model),
ident (label). 'Wrong' (W) for an answered question, per 07:
  (a) truth outside the 90% interval, or
  (b) truly not identifiable and the model answered, or
  (c) |tau| > 0.1 and the sign of the median is wrong.
"""
from __future__ import annotations

import numpy as np


def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    pos = pos.astype(bool)
    if pos.all() or (~pos).all():
        return float("nan")
    r = score.argsort().argsort().astype(np.float64) + 1
    # average ranks for ties
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wrong(tau, q05, q50, q95, ident):
    out = (tau < q05) | (tau > q95)
    out |= ident < 0.5
    out |= (np.abs(tau) > 0.1) & (np.sign(q50) != np.sign(tau))
    return out


def aurc(conf, w):
    o = np.argsort(-conf)
    risk = np.cumsum(w[o]) / np.arange(1, len(w) + 1)
    return float(risk.mean())


def report(tau, q05, q50, q95, p_id, ident, threshold=0.5, family=None) -> dict:
    tau, q05, q50, q95, p_id, ident = map(np.asarray, (tau, q05, q50, q95, p_id, ident))
    w = wrong(tau, q05, q50, q95, ident)
    width = q95 - q05
    conf = p_id / (1.0 + width)
    ans = p_id >= threshold
    idm = ident > 0.5
    oracle = aurc(-w.astype(float), w)
    e = q50 - tau
    out = {
        "n": int(len(tau)),
        "T1_AURC": aurc(conf, w),
        "T2_excess_AURC": aurc(conf, w) - oracle,
        "T3_confident_wrong@op": float(w[ans].mean()) if ans.any() else float("nan"),
        "coverage@op": float(ans.mean()),
        "T4a_nonid_AUROC": auroc(1 - p_id, ~idm),
        "CI2_said_identified_when_not": float((p_id[~idm] >= threshold).mean()) if (~idm).any() else float("nan"),
        "T5_over_abstention": float((p_id[idm] < threshold).mean()) if idm.any() else float("nan"),
        "CE1_nRMSE_identified": float(np.sqrt(np.mean(e[idm] ** 2))),
        "CE1_nMAE_identified": float(np.median(np.abs(e[idm]))),
        "CE4_coverage90_identified": float(((tau >= q05) & (tau <= q95))[idm].mean()),
        "CE5_width90_identified": float(width[idm].mean()),
        "CE6_sign_acc": float((np.sign(q50) == np.sign(tau))[idm & (np.abs(tau) > 0.1)].mean()),
    }
    if family is not None:
        fam = np.asarray(family)
        per = {}
        for f in np.unique(fam):
            m = fam == f
            if m.sum() >= 20:
                per[f] = aurc(conf[m], w[m])
        out["G1_worst_family_AURC"] = max(per.values()) if per else float("nan")
        out["G1_worst_family"] = max(per, key=per.get) if per else ""
    return out
