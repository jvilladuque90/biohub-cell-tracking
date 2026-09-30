"""Compare two sets of raw post-ILP graphs (e.g. standard detection vs. a third seed) under the
same repair (base, or without motion relink), crop by crop: score, J, adj, emitted nodes and edge
TP/FN/FP. Crops scored = those with a raw JSON in BOTH directories.

Usage:
  PYTHONPATH=src python bench/compare_raw.py data/raw_a data/raw_b --zip biohub-cell-tracking-during-development.zip \
      --public-module path/to/public_notebook_repair.py [--no-relink]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from faithful_bench import GATE, add_common_args, list_raw, load_ground_truth, run_arm, score_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Compare two raw-graph directories under the same public repair.")
    ap.add_argument("a", help="raw graph directory A (<dataset_id>.json)")
    ap.add_argument("b", help="raw graph directory B (<dataset_id>.json)")
    ap.add_argument("--no-relink", action="store_true", help="disable the motion relink step in both")
    add_common_args(ap)
    a = ap.parse_args()

    from multiprocessing import Pool

    from cellmot.evaluate import build_pred_graph
    from cellmot.io import SCALE_ZYX
    from cellmot.official.metrics import evaluate as official_evaluate
    from cellmot.official.metrics import per_sample_metrics

    scale = tuple(float(v) for v in SCALE_ZYX)
    dids = sorted(set(list_raw(a.a)) & set(list_raw(a.b)))
    gts, census = load_ground_truth(a.zip, dids, a.split)
    overrides = {"OUTPUT_MOTION_RELINK": False} if a.no_relink else {}
    tasks = [(d, raw, overrides, raw, a.public_module) for raw in (a.a, a.b) for d in dids]
    res = {}
    with Pool(a.procs) as p:
        for did, name, nodes, edges in p.imap_unordered(run_arm, tasks):
            er = official_evaluate(build_pred_graph(nodes, edges), gts[did], scale=scale, max_distance=GATE)
            f = per_sample_metrics(er, census[did], 0.0)
            sc, s1 = score_rows([f])
            res[(name, did)] = (sc, s1["edge_jaccard"], s1["adj_edge_jaccard"], len(nodes),
                                er.edge_tp, er.edge_fn, er.edge_fp, f)
    print(f"{'crop':16s} {'score A':>8s} {'score B':>8s} {'delta':>8s} | {'J A':>7s} {'J B':>7s} | "
          f"{'nodes A':>8s} {'nodes B':>8s} | {'TP/FN/FP A':>14s} {'TP/FN/FP B':>14s}")
    b_wins = 0
    for d in dids:
        ra, rb = res[(a.a, d)], res[(a.b, d)]
        b_wins += rb[0] > ra[0]
        print(f"{d:16s} {ra[0]:8.4f} {rb[0]:8.4f} {rb[0] - ra[0]:+8.4f} | {ra[1]:7.4f} {rb[1]:7.4f} | "
              f"{ra[3]:8d} {rb[3]:8d} | {ra[4]:5d}/{ra[5]:<4d}/{ra[6]:<4d} {rb[4]:5d}/{rb[5]:<4d}/{rb[6]:<4d}")
    for raw in (a.a, a.b):
        sc, s = score_rows([res[(raw, d)][7] for d in dids])
        dj = s["division_jaccard"]
        dj = 0.0 if dj != dj else dj
        print(f"  {raw:18s} SCORE {sc:.4f} J {s['edge_jaccard']:.4f} adj {s['adj_edge_jaccard']:.4f} "
              f"divJ {dj:.3f} divTP {s['division_tp']} FP {s['division_fp']} | "
              f"nodes {sum(res[(raw, d)][3] for d in dids)} TP {sum(res[(raw, d)][4] for d in dids)} "
              f"FN {sum(res[(raw, d)][5] for d in dids)} FP {sum(res[(raw, d)][6] for d in dids)}")
    print(f"  B wins on {b_wins} of {len(dids)} crops")


if __name__ == "__main__":
    main()
