"""Local evaluation with the OFFICIAL competition metric (vendored in ``cellmot.official``).

Builds tracksdata graphs and calls ``official.metrics.evaluate`` +
``per_sample_metrics`` + ``summarise`` exactly as the organizers' evaluation
script does, so the local score matches the leaderboard computation.

- Ground truth: the (tiny) ``.geff`` is extracted from the zip to a temp folder
  and loaded with ``from_geff``.
- Prediction: a tracksdata graph is built from node/edge arrays.
- Physical scale (µm) and the 7.0 µm matching threshold follow the competition.

Usage:
    from cellmot.evaluate import evaluate_prediction, load_gt_graph, self_test
    self_test("44b6_12dfb391")           # sanity check: GT against itself
"""
from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import tracksdata as td

from .io import SCALE_ZYX  # np.array([1.625, 0.40625, 0.40625])
from .official.metrics import evaluate as official_evaluate
from .official.metrics import node_recall, per_sample_metrics, summarise

SCALE_TUPLE = tuple(float(v) for v in SCALE_ZYX)   # (z, y, x) µm
MAX_DISTANCE_UM = 7.0
K = td.DEFAULT_ATTR_KEYS
DEFAULT_ZIP = "biohub-cell-tracking-during-development.zip"


# --------------------------------------------------------------------------- #
# Ground truth: extract the geff (labels only, a few KB) - never the images
# --------------------------------------------------------------------------- #
def _extract_geff(zip_path: str | Path, did: str, split: str, dest: Path) -> Path:
    pref = f"{split}/{did}.geff/"
    with zipfile.ZipFile(str(zip_path)) as zf:
        members = [n for n in zf.namelist() if n.startswith(pref)]
        if not members:
            raise FileNotFoundError(f"No geff for {split}/{did} in the zip")
        zf.extractall(dest, members=members)
    return dest / split / f"{did}.geff"


def load_gt_graph(zip_path: str | Path, did: str, split: str = "train",
                  tmpdir: Path | None = None):
    """Load the ground-truth tracksdata graph from the geff inside the zip."""
    tmp = Path(tmpdir or tempfile.mkdtemp(prefix="geff_"))
    geff_path = _extract_geff(zip_path, did, split, tmp)
    res = td.graph.IndexedRXGraph.from_geff(str(geff_path))
    return res[0] if isinstance(res, tuple) else res


def read_estimated_n_total(zip_path: str | Path, did: str,
                           split: str = "train") -> float:
    """Read ``estimated_number_of_nodes`` from the geff metadata (``N_total``).

    The official metric uses THIS value (not the actual number of GT nodes) as
    the denominator of the node-count penalty. Returns NaN if unavailable.
    """
    key = f"{split}/{did}.geff/zarr.json"
    try:
        p = Path(zip_path)
        if p.is_dir():                      # extracted folder
            meta = json.loads((p / split / f"{did}.geff" / "zarr.json").read_text())
        else:
            with zipfile.ZipFile(str(zip_path)) as zf:
                meta = json.loads(zf.read(key))
        extra = (meta.get("attributes", {}).get("geff", {}).get("extra", {}) or {})
        val = extra.get("estimated_number_of_nodes")
        return float(val) if val is not None else float("nan")
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Prediction graph construction
# --------------------------------------------------------------------------- #
def build_pred_graph(nodes: list[dict], edges: list[tuple[int, int]]):
    """Build a tracksdata graph.

    nodes: ``[{'node_id', 't', 'z', 'y', 'x'}, ...]`` with z/y/x in pixel units;
    edges: ``[(source_node_id, target_node_id), ...]``. Edges referencing unknown
    node ids are ignored.
    """
    import polars as pl
    g = td.graph.IndexedRXGraph()
    existing = set(g.node_attr_keys())
    dtypes = {K.T: (pl.Int64, 0), K.Z: (pl.Float64, 0.0),
              K.Y: (pl.Float64, 0.0), K.X: (pl.Float64, 0.0)}
    for key, (dt, dv) in dtypes.items():
        if key not in existing:
            g.add_node_attr_key(key, dtype=dt, default_value=dv)
    idx: dict[int, int] = {}
    for n in nodes:
        idx[int(n["node_id"])] = g.add_node(
            {K.T: int(n["t"]), K.Z: float(n["z"]),
             K.Y: float(n["y"]), K.X: float(n["x"])}
        )
    for s, t in edges:
        if int(s) in idx and int(t) in idx:
            g.add_edge(idx[int(s)], idx[int(t)], {})
    return g


def build_gt_like_pred(gt_graph) -> tuple[list[dict], list[tuple[int, int]]]:
    """Convert a GT graph into (nodes, edges) lists, e.g. to score prediction == GT."""
    na = gt_graph.node_attrs(attr_keys=[K.NODE_ID, K.T, K.Z, K.Y, K.X])
    nodes = [{"node_id": r[K.NODE_ID], "t": r[K.T], "z": r[K.Z],
              "y": r[K.Y], "x": r[K.X]} for r in na.iter_rows(named=True)]
    ea = gt_graph.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET])
    # EDGE_SOURCE/TARGET are internal indices; map them back to node_id
    id_by_index = {r_i: r[K.NODE_ID] for r_i, r in
                   zip(gt_graph.node_ids(), na.iter_rows(named=True))}
    edges = [(id_by_index[r[K.EDGE_SOURCE]], id_by_index[r[K.EDGE_TARGET]])
             for r in ea.iter_rows(named=True)]
    return nodes, edges


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def evaluate_graphs(pred, gt, n_total: float) -> dict:
    """Score a prediction graph against a GT graph -> one per-sample metric row.

    ``n_total`` is the target node count used by the adjusted edge Jaccard
    ``J_adj = J * (1 - 0.1 * (N_pred - N_total) / N_total)``.
    """
    er = official_evaluate(pred, gt, scale=SCALE_TUPLE, max_distance=MAX_DISTANCE_UM)
    recall = (node_recall(pred, gt)
              if pred.num_edges() > 0 and pred.num_nodes() > 0 else 0.0)
    return per_sample_metrics(er, n_total, recall)


def evaluate_prediction(zip_path: str | Path, did: str,
                        pred_nodes: list[dict], pred_edges: list[tuple[int, int]],
                        split: str = "train") -> dict:
    """Score a prediction for one dataset of the competition zip -> per-sample row."""
    gt = load_gt_graph(zip_path, did, split)
    pred = build_pred_graph(pred_nodes, pred_edges)
    row = evaluate_graphs(pred, gt, read_estimated_n_total(zip_path, did, split))
    row["dataset"] = did
    return row


def score_rows(rows: list[dict]) -> dict:
    """Aggregate per-sample rows into the run-level score (official ``summarise``)."""
    return summarise(rows)


# --------------------------------------------------------------------------- #
# Metric self-test: prediction == GT must give edge_jaccard == 1.0
# --------------------------------------------------------------------------- #
def self_test(did: str = "44b6_12dfb391", zip_path: str | Path = DEFAULT_ZIP) -> dict:
    gt = load_gt_graph(zip_path, did)
    nodes, edges = build_gt_like_pred(gt)
    row = evaluate_prediction(zip_path, did, nodes, edges)
    print(f"[self-test {did}] edge_jaccard={row['edge_jaccard']:.4f} "
          f"(TP={row['edge_tp']} FP={row['edge_fp']} FN={row['edge_fn']}) "
          f"div TP/FP/FN={row['division_tp']}/{row['division_fp']}/{row['division_fn']} "
          f"node_recall={row['node_recall']:.3f}")
    return row


if __name__ == "__main__":
    self_test()
