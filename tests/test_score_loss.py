"""CPU tests of cellmot.score_loss on synthetic data. All must pass before spending GPU time.

Usage (from the repo root): PYTHONPATH=src python -m pytest tests/test_score_loss.py -q
"""
from __future__ import annotations

import numpy as np
import torch

from cellmot import score_loss as S


def test_optimal_matching():
    """Optimal one-to-one matching at 7 um."""
    # greedy would match detection 0 to GT 0 (1 um) and leave detection 1 unmatched (GT 1 at
    # 7.5 um); the optimum crosses: 0->1 (5 um) and 1->0 (5 um) and matches both
    det = np.array([[0.0, 0, 0], [0, 0, 10.0]])
    gt = np.array([[0.0, 0, 5.0], [0, 0, -1.0]])
    m = S.match_detections(det, gt)
    assert sorted(m.tolist()) == [0, 1], f"matches both (optimal, not greedy): {m.tolist()}"
    m2 = S.match_detections(np.array([[0.0, 0, 0]]), np.array([[0.0, 0, 7.5]]))
    assert m2.tolist() == [-1], "respects the 7 um gate"


def test_prob_two_daughters():
    """Probability of >= 2 daughters."""
    e = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    p = S.prob_two_daughters(e)
    assert abs(p[0] - 1) < 1e-3 and abs(p[1]) < 1e-3 and abs(p[2] - 0.25) < 1e-3, \
        f"cases 1 / 0 / 0.25: {[round(float(x), 3) for x in p]}"


def test_fp_rule_matches_official_metric():
    """The loss's FP rule must give the SAME TP/FP as the official metric."""
    import polars as pl
    import tracksdata as td

    from cellmot.evaluate import build_pred_graph
    from cellmot.official.metrics import evaluate

    # GT: A0 -> A1 (a track); C0 is an annotated track end (no successor); D1 an annotated track
    # start (no predecessor)
    gt_nodes = {1: (0, 10, 10, 10), 2: (1, 10, 10, 12),     # A0 -> A1
                3: (0, 10, 40, 40),                           # C0 annotated, track end (no successor)
                4: (1, 10, 70, 70)}                           # D1 annotated, track start (no predecessor)
    gt_edges = [(1, 2)]
    G = td.graph.IndexedRXGraph()
    for k, dt in {td.DEFAULT_ATTR_KEYS.T: (pl.Int64, 0), "z": (pl.Float64, 0.0),
                  "y": (pl.Float64, 0.0), "x": (pl.Float64, 0.0)}.items():
        if k not in G.node_attr_keys():
            G.add_node_attr_key(k, dtype=dt[0], default_value=dt[1])
    idg = {}
    for n, (t, z, y, x) in gt_nodes.items():
        idg[n] = G.add_node({td.DEFAULT_ATTR_KEYS.T: t, "z": float(z), "y": float(y), "x": float(x)})
    for a, b in gt_edges:
        G.add_edge(idg[a], idg[b], {})
    # prediction: detections on A0, A1, C0, D1 and a spurious X1 far away
    pred = [dict(node_id=1, t=0, z=10, y=10, x=10), dict(node_id=2, t=1, z=10, y=10, x=12),
            dict(node_id=3, t=0, z=10, y=40, x=40), dict(node_id=4, t=1, z=10, y=70, x=70),
            dict(node_id=5, t=1, z=10, y=40, x=44)]
    # edges: A0->A1 (TP), C0->X1 (leaves an annotated cell WITHOUT successor -> the metric does NOT
    #        count it), A0->D1 (enters an annotated cell WITHOUT predecessor, leaves one WITH a
    #        successor -> FP)
    edges = [(1, 2), (3, 5), (1, 4)]
    er = evaluate(build_pred_graph(pred, edges), G, scale=(1.0, 1.0, 1.0), max_distance=7.0)
    # the same situation in the loss's terms (hard e = 0/1)
    # sources (t=0): A0, C0 ; targets (t=1): A1, D1, X1
    e = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out_valid = torch.tensor([True, False])          # A0 has a successor; C0 does not
    in_valid = torch.tensor([True, False, False])    # A1 has a predecessor; D1 does not; X1 not annotated
    tp, fp, _, _ = S.pair_terms(e, y, out_valid, in_valid, torch.tensor([False, False]),
                                torch.tensor([True, True]))
    assert int(tp) == er.edge_tp and int(round(float(fp))) == er.edge_fp, \
        f"TP and FP equal to the official metric (loss TP={int(tp)} FP={float(fp):.0f} | " \
        f"official TP={er.edge_tp} FP={er.edge_fp})"
    # and the OLD rule ("touches an annotated cell") would have counted C0->X1 as FP
    old = float((e * torch.tensor([[1.0, 1, 1], [1, 1, 1]]) * (1 - y)).sum())
    assert old > er.edge_fp, f"the old rule over-counts (the audited bug): old FP={old:.0f} vs {er.edge_fp}"


def test_soft_equals_hard():
    """With confident logits, the soft J equals the decoder's J."""
    torch.manual_seed(0)
    ns, nt = 6, 7
    truth = [(0, 0), (1, 1), (2, 2), (2, 3), (4, 5)]    # 2 divides; 3 and 5 have no daughters
    L = torch.full((ns, nt), -20.0)
    for i, j in truth:
        L[i, j] = 20.0
    null = torch.full((nt,), -20.0)
    null[4] = 20.0; null[6] = 20.0                        # targets 4 and 6 start a track
    e = S.parent_probs(L, null)
    y = torch.zeros(ns, nt)
    for i, j in truth:
        y[i, j] = 1
    t = torch.ones(ns, dtype=torch.bool)
    tp, fp, tpd, fpd = S.pair_terms(e, y, t, torch.ones(nt, dtype=torch.bool),
                                    torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.bool), t)
    dec = S.decode(L, null)
    assert abs(float(tp) - 5) < 1e-3 and float(fp) < 1e-3, f"soft J = 1 (TP={float(tp):.3f} FP={float(fp):.4f})"
    assert sorted((i, j) for i, j, _ in dec) == sorted(truth), "the decoder returns the truth"
    assert abs(float(tpd) - 1) < 1e-3 and float(fpd) < 1e-3, f"the division is counted (TPd={float(tpd):.3f})"


def test_gradient_raises_score():
    """The gradient raises the score."""
    L = torch.zeros(3, 3, requires_grad=True)
    null = torch.zeros(3, requires_grad=True)
    y = torch.eye(3)
    v = torch.ones(3, dtype=torch.bool)
    acc = S.new_accumulator("cpu")
    tp, fp, tpd, fpd = S.pair_terms(S.parent_probs(L, null), y, v, v, ~v, v)
    acc.update(tp=tp, fp=fp, ngt=3.0)
    r = S.soft_score(acc)
    (-r["score"]).backward()
    diag = float(torch.diagonal(L.grad).mean())
    off = float((L.grad.sum() - torch.diagonal(L.grad).sum()) / 6)
    assert diag < 0 < off or (diag < off), \
        f"pushes true edges up and false ones down (grad diag {diag:+.4f}, off {off:+.4f})"
    assert torch.isfinite(L.grad).all() and torch.isfinite(null.grad).all(), "finite gradients"


def test_count_factor():
    """Count factor without relu (as in the metric)."""
    acc = S.new_accumulator("cpu")
    acc.update(tp=torch.tensor(1.0), fp=torch.tensor(0.0), ngt=1.0,
               counts=[(torch.tensor(90.0), 100.0)])
    assert abs(float(S.soft_score(acc)["factor"]) - 1.01) < 1e-6, "emitting 90 of 100 -> 1.01"
    acc["counts"] = [(torch.tensor(110.0), 100.0)]
    assert abs(float(S.soft_score(acc)["factor"]) - 0.99) < 1e-6, "emitting 110 of 100 -> 0.99"
    # with FAR too many cells the factor is negative: the score must still increase with J
    n = torch.tensor(3000.0, requires_grad=True)
    tp = torch.tensor(0.5, requires_grad=True)
    acc2 = S.new_accumulator("cpu")
    acc2.update(tp=tp, fp=torch.tensor(0.0), ngt=1.0, counts=[(n, 100.0)])
    r = S.soft_score(acc2)
    (r["score"] - 0.1 * r["count_dev"]).backward()
    assert float(r["factor"]) < 0 and tp.grad > 0, \
        f"with a negative factor, raising J still helps (factor {float(r['factor']):.2f}, dScore/dTP {float(tp.grad):+.3f})"
    assert n.grad < 0, f"and the count term pushes to emit fewer (d/dn {float(n.grad):+.5f})"


def test_physical_features():
    """Physical features: brightness, derivative and motion."""
    Z = Y = X = 24
    zz, yy, xx = np.indices((Z, Y, X))

    def blob(cx, amp):
        return amp * np.exp(-((zz - 12) ** 2 + (yy - 12) ** 2 + (xx - cx) ** 2) / 8.0)

    v = np.stack([blob(10, 1.0), blob(12, 1.5)])            # moves +2 in x and brightens
    imgs = torch.tensor(v[None], dtype=torch.float16)
    m = S.physical_maps(imgs, 1)
    c = m[0, :, 12, 12, 12]
    assert float(c[0]) > 0.5, f"brightness at the center ({float(c[0]):.3f})"
    assert float(c[3]) > 0, f"the derivative detects brightening ({float(c[3]):+.3f})"
    assert float(c[4]) == 1.0, "flags that a previous frame exists"
    assert float(c[7]) > 0.3 and abs(float(c[5])) < 0.2 and abs(float(c[6])) < 0.2, \
        f"motion shows up in x, not in z / y (dz {float(c[5]):+.2f} dy {float(c[6]):+.2f} dx {float(c[7]):+.2f})"
    m0 = S.physical_maps(imgs, 0)
    assert float(m0[0, 4].max()) == 0.0, "without a previous frame the flag is 0"


def test_load_expanding():
    """Widening the input layer leaves the model IDENTICAL."""
    torch.manual_seed(1)
    old = torch.nn.Sequential(torch.nn.Linear(10, 4), torch.nn.LayerNorm(4))
    new = torch.nn.Sequential(torch.nn.Linear(10 + S.N_PHYS, 4), torch.nn.LayerNorm(4))
    info = S.load_expanding(new, dict(old.state_dict()))
    x = torch.randn(5, 10)
    extra = torch.randn(5, S.N_PHYS) * 100
    d = float((old(x) - new(torch.cat([x, extra], 1))).abs().max())
    assert d < 1e-5, f"same output for any value of the new features (max diff {d:.2e}; {info['expanded']})"


def test_soft_count():
    """Soft cell count."""
    L = torch.full((1, 1, 16, 16, 16), -10.0)
    L[0, 0, 4, 4, 4] = 5.0
    L[0, 0, 12, 12, 12] = 5.0
    n = S.soft_count(L, (3, 3, 3))
    assert abs(float(n[0]) - 2) < 0.02, f"two clear peaks -> ~2 ({float(n[0]):.3f})"


def test_accumulate_pair():
    """accumulate_pair with matching and the null row last."""
    # 1 sample; frame t: 2 detections on the 2 annotated cells; t+1: 3 detections, 2 on annotated cells
    det_t = (torch.tensor([[[0.0, 0, 0], [0, 0, 10]]]), None, torch.tensor([[True, True]]), None, None)
    det_t1 = (torch.tensor([[[0.0, 0, 1], [0, 0, 11], [0, 0, 30]]]), None,
              torch.tensor([[True, True, True]]), None, None)
    gt_t = torch.tensor([[[0.0, 0, 0], [0, 0, 10]]])
    gt_t1 = torch.tensor([[[0.0, 0, 11], [0, 0, 1]]])       # different order: matching fixes it
    trans = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])          # gt0 -> gt1 col 1 (x=1), gt1 -> col 0 (x=11)
    L = torch.full((1, 3, 3 + S.N_DAUGHTER_CLASSES), -20.0)   # 2 real rows + 1 null; 3 daughter columns
    L[0, 0, 0] = 20.0; L[0, 1, 1] = 20.0; L[0, 2, 2] = 20.0   # det0->det0', det1->det1', det2' no parent
    L[0, :2, 3 + 1] = 20.0                                    # daughter head: det0 and det1 with 1 daughter
    acc = S.new_accumulator("cpu")
    S.accumulate_pair(acc, L, det_t, det_t1, gt_t, gt_t1, torch.tensor([[True, True]]),
                      torch.tensor([[True, True]]), trans, (1.0, 1.0, 1.0))
    r = S.soft_score(acc)
    assert abs(float(acc["tp"]) - 2) < 1e-3 and float(acc["fp"]) < 1e-3 and acc["ngt"] == 2.0, \
        f"both true edges, no false one (TP={float(acc['tp']):.2f} FP={float(acc['fp']):.3f} NGT={acc['ngt']})"
    assert abs(float(r["J"]) - 1) < 1e-3, "J = 1"
    assert float(r["ce_daughters"]) < 1e-3, \
        f"the daughter head gets 1 daughter right for both (ce {float(r['ce_daughters']):.2e}, n {acc['n_daughters']})"


def test_joint_assignment():
    """Joint assignment: a parent does not keep two daughters unless the model predicts a division."""
    # targets 0 and 1 BOTH prefer parent 0; the true parent of 1 is 1 (second choice)
    L = torch.tensor([[3.0, 2.0], [-5.0, 1.0]])
    null = torch.tensor([-9.0, -9.0])
    free = sorted((i, j) for i, j, _ in S.decode(L, null))
    assert free == [(0, 0), (0, 1)], f"without the daughter head both pick parent 0 (false division): {free}"
    one = torch.tensor([[-9.0, 9.0, -9.0], [-9.0, 9.0, -9.0]])      # both parents: 1 daughter
    joint = sorted((i, j) for i, j, _ in S.decode(L, null, one))
    assert joint == [(0, 0), (1, 1)], f"with the head (1 daughter) parent 1 gets its daughter back: {joint}"
    two = torch.tensor([[-9.0, -9.0, 9.0], [9.0, -9.0, -9.0]])      # parent 0 divides, parent 1 none
    joint2 = sorted((i, j) for i, j, _ in S.decode(L, null, two))
    assert joint2 == [(0, 0), (0, 1)], f"with the head (2 daughters) the division is kept: {joint2}"
    # "no parent" wins when the parent is taken and the alternative is bad
    L3 = torch.tensor([[3.0, 2.0], [-9.0, -9.0]])
    null3 = torch.tensor([-9.0, 0.0])
    zero = torch.tensor([[-9.0, 9.0, -9.0], [9.0, -9.0, -9.0]])     # parent 1: no daughters
    joint3 = sorted((i, j) for i, j, _ in S.decode(L3, null3, zero))
    assert joint3 == [(0, 0)], f"if the only possible parent already has its daughter, the other starts a track: {joint3}"
    # the gradient reaches the daughter head through the soft division term
    h = torch.zeros(2, S.N_DAUGHTER_CLASSES, requires_grad=True)
    e = S.parent_probs(L, null)
    v = torch.ones(2, dtype=torch.bool)
    _, _, tpd, fpd = S.pair_terms(e, torch.eye(2), v, v, torch.tensor([True, False]), v,
                                  p2=torch.softmax(h, -1)[:, 2])
    (-(tpd - fpd)).backward()
    assert float(h.grad[0, 2]) < 0 < float(h.grad[1, 2]), \
        f"the soft division raises P(2 daughters) where there is one and lowers it where not " \
        f"(grad {float(h.grad[0, 2]):+.3f} / {float(h.grad[1, 2]):+.3f})"


def test_division_memory():
    """Division Jaccard with memory: a batch WITHOUT divisions also penalizes false ones."""
    def batch(p2_logit, has_div):
        h = torch.tensor([[0.0, 0.0, p2_logit]], requires_grad=True)
        p2 = torch.softmax(h, -1)[:, 2]
        div = torch.tensor([has_div])
        acc = S.new_accumulator("cpu")
        acc.update(tp=torch.tensor(1.0), fp=torch.tensor(0.0), ngt=1.0,
                   tpd=(p2 * div.float()).sum(), fpd=(p2 * (~div).float()).sum(), nd=int(has_div))
        return h, acc

    h, acc = batch(0.0, False)
    r = S.soft_score(acc)
    assert (not r["divJ"].requires_grad
            or float(torch.autograd.grad(r["divJ"], h, allow_unused=True)[0] is None or 0) == 0), \
        "WITHOUT memory: a batch without divisions gives no gradient to the division term"
    mem = S.DivisionMemory()
    h1, acc1 = batch(3.0, True)                      # an earlier batch with a well-caught division
    S.soft_score(acc1, mem)
    h2, acc2 = batch(0.0, False)                     # current batch: no divisions, P(2 daughters) 1/3
    r2 = S.soft_score(acc2, mem)
    (-r2["divJ"]).backward()
    assert float(h2.grad[0, 2]) > 0, \
        f"WITH memory: the gradient lowers P(2 daughters) of a non-dividing cell " \
        f"(grad {float(h2.grad[0, 2]):+.4f}, divJ {float(r2['divJ']):.3f})"
    assert mem.nd > 0.9, f"the memory remembers the previous batch's division (nd {mem.nd:.3f})"


def test_exclusivity():
    """Exclusivity: one detection per annotated cell."""
    vox = (1.625, 1.625, 1.625)
    gt = torch.tensor([[[8.0, 8, 8]]]); m = torch.tensor([[True]])
    single = torch.full((1, 1, 16, 16, 16), -10.0); single[0, 0, 8, 8, 8] = 6.0
    assert float(S.exclusivity(single, gt, m, (3, 3, 3), vox)) < 0.01, "one peak on the annotated cell -> ~0"
    dup = single.clone(); dup[0, 0, 8, 8, 11] = 6.0                                  # another peak at 4.9 um
    L = dup.clone().requires_grad_(True)
    v = S.exclusivity(L, gt, m, (3, 3, 3), vox)
    assert abs(float(v) - 1) < 0.02, f"two peaks on the same annotated cell -> ~1 ({float(v):.3f})"
    v.backward()
    assert float(L.grad[0, 0, 8, 8, 11]) > 0, \
        f"the gradient lowers the duplicate peak ({float(L.grad[0, 0, 8, 8, 11]):+.4f})"
    far = single.clone(); far[0, 0, 8, 8, 15] = 6.0                                  # at 11.4 um: another cell
    assert float(S.exclusivity(far, gt, m, (3, 3, 3), vox)) < 0.01, "a peak > 7 um away is not a duplicate"
    empty = torch.full((1, 1, 16, 16, 16), -10.0)
    assert float(S.exclusivity(empty, gt, m, (3, 3, 3), vox)) > 0.9, "annotated cell without detection -> ~1"


def test_video_level_divisions():
    """Divisions chosen for the whole video: a chosen cell takes two daughters, the rest one."""
    L = torch.tensor([[3.0, 2.0], [-5.0, -5.0]])       # both daughters prefer parent 0
    null = torch.tensor([-9.0, 0.0])
    one = torch.tensor([[-9.0, 9.0, -9.0], [9.0, -9.0, -9.0]])
    no = sorted((i, j) for i, j, _ in S.decode(L, null, one, two=np.array([False, False])))
    yes = sorted((i, j) for i, j, _ in S.decode(L, null, one, two=np.array([True, False])))
    assert no == [(0, 0)], f"not chosen: parent 0 keeps one daughter, the other starts a track: {no}"
    assert yes == [(0, 0), (0, 1)], f"chosen: parent 0 divides even though the head said 1 daughter: {yes}"
