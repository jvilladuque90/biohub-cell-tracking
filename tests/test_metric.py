"""Sanity tests of the official metric wrapper on tiny synthetic graphs (no competition data).

Coordinates are in pixels; the metric converts them to µm with the anisotropic
voxel scale (z=1.625, y=x=0.40625 µm) and matches centroids within 7 µm.
"""
from __future__ import annotations

import warnings

import pytest

from cellmot.evaluate import build_pred_graph, evaluate_graphs, score_rows

# Two lineages placed ~40 µm apart in x so they can never be confused.
#   A: a0 -> a1 -> a2 (divides) -> {b3 -> b4, c3 -> c4}
#   B: d0 -> d1 -> d2 -> d3 -> d4
A_X, B_X = 50.0, 150.0


def _lineage_a():
    nodes = [
        {"node_id": 1, "t": 0, "z": 10, "y": 50, "x": A_X},
        {"node_id": 2, "t": 1, "z": 10, "y": 50, "x": A_X},
        {"node_id": 3, "t": 2, "z": 10, "y": 50, "x": A_X},
        {"node_id": 4, "t": 3, "z": 10, "y": 40, "x": A_X},
        {"node_id": 5, "t": 3, "z": 10, "y": 60, "x": A_X},
        {"node_id": 6, "t": 4, "z": 10, "y": 35, "x": A_X},
        {"node_id": 7, "t": 4, "z": 10, "y": 65, "x": A_X},
    ]
    edges = [(1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 7)]
    return nodes, edges


def _lineage_b():
    nodes = [{"node_id": 10 + t, "t": t, "z": 10, "y": 50, "x": B_X} for t in range(5)]
    edges = [(10 + t, 11 + t) for t in range(4)]
    return nodes, edges


def _gt(with_division: bool = True):
    nb, eb = _lineage_b()
    if not with_division:
        return nb, eb
    na, ea = _lineage_a()
    return na + nb, ea + eb


def _score(pred_nodes, pred_edges, gt_nodes, gt_edges, n_total=None):
    gt = build_pred_graph(gt_nodes, gt_edges)
    pred = build_pred_graph(pred_nodes, pred_edges)
    n_total = float(len(gt_nodes)) if n_total is None else n_total
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = evaluate_graphs(pred, gt, n_total)
        summary = score_rows([row])
    return row, summary


def _shift(nodes, node_id, **offsets):
    out = []
    for n in nodes:
        n = dict(n)
        if n["node_id"] == node_id:
            for k, v in offsets.items():
                n[k] = n[k] + v
        out.append(n)
    return out


def test_identity_without_division_scores_one():
    nodes, edges = _gt(with_division=False)
    row, summary = _score(nodes, edges, nodes, edges)
    assert row["edge_jaccard"] == pytest.approx(1.0)
    assert row["adj_edge_jaccard"] == pytest.approx(1.0)
    assert summary["score"] == pytest.approx(1.0)


def test_identity_with_division():
    nodes, edges = _gt()
    row, summary = _score(nodes, edges, nodes, edges)
    assert (row["edge_tp"], row["edge_fp"], row["edge_fn"]) == (len(edges), 0, 0)
    assert (row["division_tp"], row["division_fp"], row["division_fn"]) == (1, 0, 0)
    assert summary["edge_jaccard"] == pytest.approx(1.0)
    assert summary["division_jaccard"] == pytest.approx(1.0)
    # score = adj_edge_jaccard + 0.1 * division_jaccard (unnormalised, max 1.1)
    assert summary["score"] == pytest.approx(1.1)


def test_dropping_an_edge():
    nodes, edges = _gt()
    pred_edges = [e for e in edges if e != (12, 13)]  # break lineage B
    row, _ = _score(nodes, pred_edges, nodes, edges)
    n = len(edges)
    assert (row["edge_tp"], row["edge_fp"], row["edge_fn"]) == (n - 1, 0, 1)
    assert row["edge_jaccard"] == pytest.approx((n - 1) / n)
    assert row["division_tp"] == 1  # the division is untouched


def test_missing_division_edge_loses_division():
    nodes, edges = _gt()
    pred_edges = [e for e in edges if e != (3, 5)]
    row, summary = _score(nodes, pred_edges, nodes, edges)
    assert (row["division_tp"], row["division_fn"]) == (0, 1)
    assert summary["division_jaccard"] == pytest.approx(0.0)


def test_extra_unmatched_nodes_penalise_adjusted_jaccard():
    nodes, edges = _gt()
    n_gt = len(nodes)
    m = 4
    extra = [{"node_id": 100 + i, "t": i, "z": 10, "y": 200, "x": 250} for i in range(m)]
    row, _ = _score(nodes + extra, edges, nodes, edges)
    assert row["edge_jaccard"] == pytest.approx(1.0)  # isolated nodes add no edges
    assert row["num_pred_nodes"] == n_gt + m
    ratio = m / n_gt
    assert row["total_node_ratio"] == pytest.approx(ratio)
    assert row["adj_edge_jaccard"] == pytest.approx(1.0 - 0.1 * ratio)


def test_fewer_nodes_than_n_total_increase_adjusted_jaccard():
    # J_adj has no upper cap: emitting fewer nodes than N_total multiplies J by > 1.
    nodes, edges = _gt()
    n_total = 2.0 * len(nodes)
    row, _ = _score(nodes, edges, nodes, edges, n_total=n_total)
    assert row["adj_edge_jaccard"] == pytest.approx(1.0 * (1 - 0.1 * (-0.5)))


@pytest.mark.parametrize(
    "offset, matches",
    [
        ({"z": 4}, True),    # 4 * 1.625 = 6.5 µm
        ({"z": 5}, False),   # 5 * 1.625 = 8.125 µm
        ({"x": 17}, True),   # 17 * 0.40625 = 6.906 µm
        ({"x": 18}, False),  # 18 * 0.40625 = 7.3125 µm
        ({"y": 12, "z": 3}, True),   # sqrt(4.875^2 + 4.875^2) = 6.894 µm
        ({"y": 12, "z": 4}, False),  # sqrt(4.875^2 + 6.5^2)   = 8.125 µm
    ],
)
def test_matching_threshold_is_7um_with_anisotropic_scale(offset, matches):
    nodes, edges = _gt()
    pred_nodes = _shift(nodes, 12, **offset)  # interior node of lineage B (2 edges)
    row, _ = _score(pred_nodes, edges, nodes, edges)
    n = len(edges)
    if matches:
        assert (row["edge_tp"], row["edge_fp"], row["edge_fn"]) == (n, 0, 0)
    else:
        # both incident edges are lost and count as false positives
        assert (row["edge_tp"], row["edge_fp"], row["edge_fn"]) == (n - 2, 2, 2)
        assert row["edge_jaccard"] == pytest.approx((n - 2) / (n + 2))
