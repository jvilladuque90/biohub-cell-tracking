"""FAITHFUL BENCH: the public notebook's graph repair (`filter_output_graph`, the same code that
runs on Kaggle) applied to the RAW post-ILP graphs that the notebook itself produced on training
crops (<raw>/<dataset_id>.json: two-seed detection + TTA + 0.965 threshold + bidirectional edge
fusion + its ILP). Each arm changes ONE constant (or a group sharing one mechanism) via setattr on
the module, runs the whole repair and scores it with the official metric against the FULL ground
truth.

No DeepCenter (model not available locally) and no HOCT veto: this measures PRE-veto deltas. The
veto is worth ~+0.002 locally and +0.006 on the LB and is nearly orthogonal to these constants.

Raw graph JSON format: {"nodes": [[id, t, z, y, x], ...], "edges": [[src, tgt, prob], ...]}
(coordinates in voxels). Crops scored = those with a raw JSON.

The repair module is third-party (see public_repair.py) and is loaded lazily inside the workers.

Usage:
  PYTHONPATH=src python bench/faithful_bench.py --zip biohub-cell-tracking-during-development.zip \
      --raw data/raw --public-module path/to/public_notebook_repair.py [--procs 4] [--only base,len5]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GATE = 7.0   # official centroid matching radius (um)

ARMS = {
    "base": {},
    # pruning / rescue of short tracks (components of 4-5 nodes containing annotated cells)
    "len5": {"OUTPUT_MIN_TRACK_LEN": 5},
    "len8": {"OUTPUT_MIN_TRACK_LEN": 8},
    "rescue_len3": {"SHORT_TRACK_RESCUE_MIN_LEN": 3},
    "rescue_frac3x": {"SHORT_TRACK_RESCUE_MAX_NODES_FRAC": 0.036, "SHORT_TRACK_RESCUE_MAX_NODES_ABS": 360},
    "rescue_prob80": {"SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": 0.80},
    "no_rescue": {"ADAPTIVE_SHORT_TRACK_RESCUE": False},
    # gap closing
    "gap_um7": {"GAP_CLOSE_UM": 7.0},
    "gap_um4.5": {"GAP_CLOSE_UM": 4.5},
    "gap_only1": {"OUTPUT_GAP2_RECOVERY": False},
    "gap_no_density": {"GAP_DENSITY_ADAPTIVE": False},
    "gap_no_refine": {"GAP_REFINE_SYNTHETIC": False},
    # motion relink
    "relink_tight4.5": {"MOTION_RELINK_TIGHT_UM": 4.5},
    "relink_tight6.5": {"MOTION_RELINK_TIGHT_UM": 6.5},
    "relink_relaxed8": {"MOTION_RELINK_RELAXED_UM": 8.0},
    "relink_bonus0.5": {"MOTION_RELINK_LEARNED_BONUS": 0.5},
    "no_relink": {"OUTPUT_MOTION_RELINK": False},
    # position smoothing
    "no_smoothing": {"OUTPUT_LINEFIT_SMOOTH": False},
    "smoothing_w3": {"OUTPUT_LINEFIT_WINDOW": 3},
    "smoothing_weight0.5": {"OUTPUT_LINEFIT_WEIGHT": 0.5},
    # divisions (the LB said opening them costs score: test CLOSING)
    "div_narrow": {"SAFE_DIV_MAX_UM": 7.0, "SAFE_DIV_SISTER_MAX_UM": 11.0, "SAFE_DIV_EXISTING_CHILD_MAX_UM": 8.0},
    "div_tau0.4": {"SAFE_DIV_SISTER_SYMMETRY_TAU": 0.4},
    "div_caps_half": {"SAFE_DIV_FRAME_FRAC_CAP": 0.0038, "SAFE_DIV_GLOBAL_FRAC_CAP": 0.0019},
    "div_diverge3": {"SAFE_DIV_DIVERGE_UM": 3.0},
    "no_divisions": {"OUTPUT_SAFE_DIVISIONS": False},
    # max edge length and parent repair
    "edge_max12": {"OUTPUT_EDGE_MAX_UM": 12.0},
    "no_single_parent": {"OUTPUT_SINGLE_PARENT_REPAIR": False},
    # combinations of the arms that were positive in the first pass
    "relink_tight5.5": {"MOTION_RELINK_TIGHT_UM": 5.5},
    "relink_tight7.5": {"MOTION_RELINK_TIGHT_UM": 7.5},
    "div_close": {"SAFE_DIV_SISTER_SYMMETRY_TAU": 0.4, "SAFE_DIV_DIVERGE_UM": 3.0},
    "combo_norelink_div": {"OUTPUT_MOTION_RELINK": False, "SAFE_DIV_SISTER_SYMMETRY_TAU": 0.4, "SAFE_DIV_DIVERGE_UM": 3.0},
    "combo_norelink_smooth": {"OUTPUT_MOTION_RELINK": False, "OUTPUT_LINEFIT_WEIGHT": 0.5},
    "combo_all": {"OUTPUT_MOTION_RELINK": False, "SAFE_DIV_SISTER_SYMMETRY_TAU": 0.4, "SAFE_DIV_DIVERGE_UM": 3.0,
                  "OUTPUT_LINEFIT_WEIGHT": 0.5, "OUTPUT_MIN_TRACK_LEN": 8},
    # no relink + let gap closing join what the relink joined correctly
    "nr_gap6": {"OUTPUT_MOTION_RELINK": False, "GAP_CLOSE_UM": 6.0},
    "nr_gap7": {"OUTPUT_MOTION_RELINK": False, "GAP_CLOSE_UM": 7.0},
    "nr_gap8": {"OUTPUT_MOTION_RELINK": False, "GAP_CLOSE_UM": 8.0},
    "nr_gap2_wide": {"OUTPUT_MOTION_RELINK": False, "GAP2_MAX_STEP_UM": 6.0, "GAP2_MAX_TOTAL_UM": 14.0},
    "nr_gap_frac": {"OUTPUT_MOTION_RELINK": False, "GAP_CLOSE_MAX_ADDED_FRAC": 0.10, "GAP_CLOSE_MAX_ADDED_ABS": 4000},
    "nr_rescue80": {"OUTPUT_MOTION_RELINK": False, "SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": 0.80},
    "nr_reuse5": {"OUTPUT_MOTION_RELINK": False, "GAP_CLOSE_REUSE_UM": 5.0},
}


def run_arm(task):
    """Worker: repair one raw graph with one arm's overrides. Everything travels in the task
    tuple (spawn-safe): (dataset_id, arm_name, overrides, raw_dir, public_module_path)."""
    did, name, overrides, raw, module_path = task
    import public_repair
    m = public_repair.load_public_module(module_path)
    with open(os.path.join(raw, did + ".json")) as fh:
        g = json.load(fh)
    nodes_by_id = {int(n[0]): {"node_id": int(n[0]), "t": int(n[1]), "z": float(n[2]), "y": float(n[3]), "x": float(n[4])}
                   for n in g.get("nodes", g.get("nodos"))}
    raw_edges = [{"source_id": int(a), "target_id": int(b), "edge_prob": p} for a, b, p in g.get("edges", g.get("aristas"))]
    saved = {k: getattr(m, k) for k in overrides}
    for k, v in overrides.items():
        setattr(m, k, v)
    try:
        nb, ee, _ = m.filter_output_graph(nodes_by_id, raw_edges, dataset=None, deepcenter_bundle=None)
    finally:
        for k, v in saved.items():
            setattr(m, k, v)
    nodes = [{"node_id": int(k), "t": int(v["t"]), "z": float(v["z"]), "y": float(v["y"]), "x": float(v["x"])}
             for k, v in sorted(nb.items())]
    edges = [(int(e["source_id"]), int(e["target_id"])) for e in ee]
    return did, name, nodes, edges


def list_raw(raw_dir):
    return sorted(f[:-5] for f in os.listdir(raw_dir) if f.endswith(".json"))


def load_ground_truth(zip_path, dids, split="train"):
    """Full GT graph (all annotated lineages) and census (`estimated_number_of_nodes`) per crop."""
    from cellmot.evaluate import load_gt_graph, read_estimated_n_total
    gts, census = {}, {}
    for d in dids:
        gts[d] = load_gt_graph(zip_path, d, split)
        census[d] = read_estimated_n_total(zip_path, d, split)
    return gts, census


def score_rows(rows):
    """Official score of a set of per-sample rows: adj_edge_jaccard + 0.1 * division_jaccard."""
    from cellmot.official.metrics import summarise
    s = summarise(list(rows))
    dj = s["division_jaccard"]
    dj = 0.0 if dj != dj else dj
    return s["adj_edge_jaccard"] + 0.1 * dj, s


def add_common_args(ap):
    ap.add_argument("--zip", default="biohub-cell-tracking-during-development.zip",
                    help="competition zip (ground truth geff is read from it)")
    ap.add_argument("--split", default="train", help="split inside the zip holding the crops")
    ap.add_argument("--public-module", default=os.environ.get("BIOHUB_PUBLIC_MODULE"),
                    help="path to the third-party repair module (default: $BIOHUB_PUBLIC_MODULE)")
    ap.add_argument("--procs", type=int, default=4, help="worker processes")


def main():
    ap = argparse.ArgumentParser(description="Faithful bench: public graph repair with overridden constants, "
                                             "scored against full ground truth.")
    add_common_args(ap)
    ap.add_argument("--raw", default="data/raw", help="directory of raw post-ILP graphs (<dataset_id>.json)")
    ap.add_argument("--only", default="", help="comma-separated subset of arms to run")
    ap.add_argument("--out", default="", help="optional JSON file for per-arm score deltas")
    a = ap.parse_args()

    from multiprocessing import Pool

    from cellmot.evaluate import build_pred_graph
    from cellmot.io import SCALE_ZYX
    from cellmot.official.metrics import evaluate as official_evaluate
    from cellmot.official.metrics import per_sample_metrics

    scale = tuple(float(v) for v in SCALE_ZYX)
    dids = list_raw(a.raw)
    gts, census = load_ground_truth(a.zip, dids, a.split)
    arms = {k: v for k, v in ARMS.items() if not a.only or k in a.only.split(",") or k == "base"}
    print(f"{len(dids)} crops | {len(arms)} arms")
    tasks = [(d, n, c, a.raw, a.public_module) for n, c in arms.items() for d in dids]
    rows = collections.defaultdict(dict)
    n_nodes = collections.Counter()
    with Pool(a.procs) as p:
        for did, name, nodes, edges in p.imap_unordered(run_arm, tasks):
            er = official_evaluate(build_pred_graph(nodes, edges), gts[did], scale=scale, max_distance=GATE)
            rows[name][did] = per_sample_metrics(er, census[did], 0.0)
            n_nodes[name] += len(nodes)

    sc_base, _ = score_rows(rows["base"].values())
    base_per_crop = {d: score_rows([f])[0] for d, f in rows["base"].items()}
    print(f"\n  {'arm':22s} {'SCORE':>7s} {'delta':>8s} {'adj':>7s} {'divJ':>6s} {'divTP/FP':>9s} {'nodes':>8s}  wins/losses per crop")
    res = {}
    for name in arms:
        sc, s = score_rows(rows[name].values())
        dj = s["division_jaccard"]
        dj = 0.0 if dj != dj else dj
        wins = sum(1 for d, f in rows[name].items() if score_rows([f])[0] > base_per_crop[d] + 1e-9)
        losses = sum(1 for d, f in rows[name].items() if score_rows([f])[0] < base_per_crop[d] - 1e-9)
        res[name] = sc - sc_base
        print(f"  {name:22s} {sc:7.4f} {sc - sc_base:+8.4f} {s['adj_edge_jaccard']:7.4f} {dj:6.3f} "
              f"{s['division_tp']:3d}/{s['division_fp']:<5d} {n_nodes[name]:8d}  {wins}/{losses}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)

    # per-crop detail for arms with |delta| >= 0.003: median, range and mean without the largest |delta|
    for name in arms:
        if name == "base" or abs(res[name]) < 0.003:
            continue
        d = {did: score_rows([rows[name][did]])[0] - base_per_crop[did] for did in rows[name]}
        v = sorted(d.values())
        worst = max(d, key=lambda k: abs(d[k]))
        rest = [x for k, x in d.items() if k != worst]
        rest_mean = statistics.mean(rest) if rest else float("nan")
        print(f"  {name}: median {statistics.median(v):+.4f} | min {v[0]:+.4f} max {v[-1]:+.4f} | "
              f"without {worst} ({d[worst]:+.4f}): mean {rest_mean:+.4f}")
        print("    " + " ".join(f"{k[-8:]}:{x:+.3f}" for k, x in sorted(d.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
