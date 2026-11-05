"""Reproduce the Piston v0.1 leaderboard on the frozen benchmark (split C).

    python evaluate_benchmark.py --weights weights --bench benchmark            # model + baselines
    python evaluate_benchmark.py --weights weights --bench benchmark --boot 2000  # + 95% intervals

Rows:
  piston | none / partial / full      graph knowledge given as input (partial: each ordered pair
                                          revealed with probability U(0.2, 0.8), fixed seed)
  hybrid                                  graph known: refuse exactly the questions labelled not
                                          identified (an exact identification algorithm), answer the
                                          rest with the model's no-graph ranges
  baselines                               from benchmark/baseline_predictions.npz (widened intervals)
To score your own method, write (q05, q50, q95, p_answerable) per question in split order and use
piston.metrics.report.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from piston.data import ShardDataset
from piston.metrics import aurc, report, wrong
from piston.predict import Piston


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--bench", default="benchmark")
    ap.add_argument("--boot", type=int, default=0, help="episode-clustered bootstrap resamples (0 = skip)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    bench = Path(a.bench)
    ds = ShardDataset([bench / "split_C.npz"])
    idx = np.arange(len(ds))
    ident = ds.ident.astype(float)
    ep = ds.q_ep
    fam = ds.family[ep]
    pst = Piston.from_pretrained(a.weights, device=a.device)
    print(f"{len(pst.models)} checkpoints; {len(idx)} questions in {len(np.unique(ep))} worlds", flush=True)

    rows = {}
    for kn in ("none", "partial", "full"):
        krng = np.random.default_rng(12345)
        out = []
        for s in range(0, len(idx), 256):
            out.append(pst.predict_batch(ds.batch(idx[s:s + 256], None, know=kn, know_rng=krng), know=kn))
        p = np.concatenate(out)
        rows[f"piston | {kn} graph"] = (p[:, 0], p[:, 1], p[:, 2], p[:, 3])
    tau = np.array([ds.batch(np.array([i]), None)["tau"][0] for i in idx])   # in the model's z-units
    lo, med, hi, _ = rows["piston | none graph"]
    rows["hybrid: exact identification + piston"] = (lo, med, hi, ident)

    bp = bench / "baseline_predictions.npz"
    if bp.exists():
        z = np.load(bp)
        for k in z.files:
            v = z[k]
            rows[k] = (v[:, 0], v[:, 1], v[:, 2], v[:, 3])

    res = {k: report(tau, *r, ident, family=fam) for k, r in rows.items()}
    if a.boot:
        eps = np.unique(ep)
        by = {e: np.where(ep == e)[0] for e in eps}
        rng = np.random.default_rng(0)
        T1 = {k: [] for k in rows}
        W = {k: wrong(tau, *rows[k][:3], ident) for k in rows}
        for _ in range(a.boot):
            ix = np.concatenate([by[e] for e in rng.choice(eps, len(eps))])
            for k, (l, _m, h, pid) in rows.items():
                T1[k].append(aurc(pid[ix] / (1 + h[ix] - l[ix]), W[k][ix]))
        for k in rows:
            res[k]["T1_CI95"] = [float(np.percentile(T1[k], 2.5)), float(np.percentile(T1[k], 97.5))]

    print(f"\n{'row':52s} {'T1':>6s} {'T3':>6s} {'answered':>8s} {'cover90':>7s} {'nMAE':>6s}" + ("   T1 95%" if a.boot else ""))
    for k, r in res.items():
        ci = f"   [{r['T1_CI95'][0]:.3f}, {r['T1_CI95'][1]:.3f}]" if a.boot else ""
        print(f"{k:52s} {r['T1_AURC']:6.3f} {r['T3_confident_wrong@op']:6.3f} {r['coverage@op']:8.2f} "
              f"{r['CE4_coverage90_identified']:7.3f} {r['CE1_nMAE_identified']:6.3f}{ci}")
    Path("benchmark_results.json").write_text(json.dumps(res, indent=1))
    print("\nsaved benchmark_results.json")


if __name__ == "__main__":
    main()
