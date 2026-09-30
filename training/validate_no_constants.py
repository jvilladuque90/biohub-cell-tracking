"""VALIDATION WITH THE OFFICIAL METRIC, decoding WITHOUT CONSTANTS (exactly what would be shipped).

Copied into the organizer's repo (scripts/) next to score_loss.py. Called by the epoch loop that
training/patch_score.py installs, with the EMA weights. It is also the inference of the new model:
  - cells: local maxima of the detection map with probability > 0.5 (the model's decision; the
    non-maximum suppression uses the radius the model was trained with, pool_kernel_um)
  - edges: each cell of the next frame picks a parent or "no parent" and each parent its number
    of daughters, solved jointly by optimal assignment (score_loss.decode)
  - nothing else: no global linker, no tuned thresholds, no graph repair.
The window slides over the video with stride W-1 (each frame pair is decided exactly once) and the
cells of a frame are fixed the first time the frame appears, so consecutive pairs share identities.

Detection knobs read from the environment:
  LOGIT_THRESHOLD  detection logit threshold (default 0 = probability 0.5); negative = more
                   recall, positive = more precision
  TTA_AGG          aggregation of the TTA views: "mean" (deployed), "max" (union: a cell seen in
                   ANY view is kept -> more recall), "min" (consensus: kept only if it clears the
                   threshold in ALL views -> more precision)
  NMS_UM           non-maximum-suppression radius in um (default: the training radius)
"""
from __future__ import annotations

import os as _os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr

try:                                   # inside the organizer's repo (scripts/ on sys.path)
    import score_loss as SL
except ImportError:                    # inside this repo (package installed from src/)
    from cellmot import score_loss as SL

LOGIT_THRESHOLD = float(_os.environ.get("LOGIT_THRESHOLD", "0.0"))
TTA_AGG = _os.environ.get("TTA_AGG", "mean")


def census_of(ds_path: Path) -> float:
    """estimated_number_of_nodes from the .geff (what the metric uses as N_total)."""
    try:
        g = zarr.open_group(str(ds_path.parent / f"{ds_path.name}.geff"), mode="r")
        return float(g.attrs["geff"]["extra"]["estimated_number_of_nodes"])
    except Exception:
        return float("nan")


def kernel_nms(pool_kernel_um: float, voxel_size) -> tuple:
    """Same computation as the organizer's detect_and_match."""
    return tuple(max(1, k if k % 2 == 1 else k + 1)
                 for k in (max(1, round(pool_kernel_um / s)) for s in voxel_size))


@torch.no_grad()
def find_peaks(det_logit: torch.Tensor, pk: tuple) -> torch.Tensor:
    """(Z, Y, X) logits -> (n, 3) coordinates of local maxima above LOGIT_THRESHOLD."""
    pad = tuple(k // 2 for k in pk)
    L = det_logit[None, None]
    is_peak = (L == F.max_pool3d(L, pk, 1, pad)) & (L > LOGIT_THRESHOLD)
    return torch.nonzero(is_peak[0, 0]).float()


def build_graph(nodes: list, edges: list):
    import polars as pl
    import tracksdata as td
    K = td.DEFAULT_ATTR_KEYS
    g = td.graph.IndexedRXGraph()
    for key, (dt, dv) in {K.T: (pl.Int64, 0), K.Z: (pl.Float64, 0.0), K.Y: (pl.Float64, 0.0),
                          K.X: (pl.Float64, 0.0)}.items():
        if key not in set(g.node_attr_keys()):
            g.add_node_attr_key(key, dtype=dt, default_value=dv)
    idx = {}
    for n in nodes:
        idx[n["node_id"]] = g.add_node({K.T: int(n["t"]), K.Z: float(n["z"]), K.Y: float(n["y"]),
                                        K.X: float(n["x"])})
    for s, t in edges:
        g.add_edge(idx[s], idx[t], {})
    LAST_IDX.clear()
    LAST_IDX.update(idx)          # our node_id -> graph id (used by the dump)
    return g


@torch.no_grad()
def infer(model, ds, device, window_size: int, pool_kernel_um: float, pos_embed_fn,
          max_frames: int | None = None):
    """Whole video -> (nodes, edges) in full-resolution voxels, without constants.
    `model` may be a single model or a list (seed ensemble)."""
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    _group = zarr.open_group(str(ds.zarr_path), mode="r")
    zg = _group["0"]
    zds = _group["ds"] if "ds" in _group else None     # pre-downsampled data (Colab)
    T = int(zg.shape[0]) if max_frames is None else min(int(zg.shape[0]), max_frames)
    # the organizer's Dataset does NOT store the downsampling factor: use the one set by the
    # training loop, the same one used to load the data
    downsample = tuple(int(v) for v in (getattr(ds, "downsample", None) or DOWNSAMPLE))
    dz, dy, dx = downsample
    shape = list(ds.image_shape[1:])
    ql, qh = float(ds.quantiles["0.001"]), float(ds.quantiles["0.999"])
    voxel = tuple(s * d for s, d in zip(ds.scale, downsample))
    # NMS_UM (environment) overrides the NMS radius (default: the training one, 5 um): in dense
    # crops two nuclei < 5 um apart merge into one peak and the annotated cell is lost
    pk = kernel_nms(float(_os.environ.get("NMS_UM", pool_kernel_um)), voxel)
    ds_t = torch.tensor(downsample, dtype=torch.float32, device=device)
    W = window_size
    pos_shape = (W,) + tuple(shape)

    nodes_t: dict[int, torch.Tensor] = {}
    det_t: dict[int, torch.Tensor] = {}      # DUMP: detection logit at each peak
    p_edge: dict = {}                        # DUMP: (t, i, t1, j) -> probability of the chosen edge
    p_daughters: dict = {}                   # DUMP: (t, i) -> softmax of the daughter head (0, 1, 2)
    done: set[int] = set()
    done_skip: set[int] = set()
    edges = []
    pairs = []
    skips = []
    # with SKIPS the window advances 1 frame at a time: every (t, t+2) pair falls entirely inside
    # some window
    step = 1 if (SKIPS and W >= 3) else W - 1
    starts = list(range(0, max(1, T - W + 1), step))
    if starts[-1] + W < T:
        starts.append(T - W)
    for s in starts:
        e = min(s + W, T)
        raw = (zds[s:e] if zds is not None else zg[s:e, ::dz, ::dy, ::dx]).astype(np.float32)
        im = torch.from_numpy((raw - ql) / (qh - ql + 1e-6)).clamp(0.0)
        if list(im.shape[1:]) != shape:
            im = F.interpolate(im[:, None], size=shape, mode="trilinear", align_corners=False)[:, 0]
        imgs = im[None].to(device)
        # SEED ENSEMBLE: each model encodes the window and their detection maps are averaged (and,
        # below, their edge logits). With a single model nothing changes.
        unet_outs, dets = [], []
        for mm in models:
            uo, dl = mm.encode(imgs)
            unet_outs.append(uo)
            dets.append(dl)
        det_logits = [sum(d[f] for d in dets) / len(dets) for f in range(len(dets[0]))]
        if TTA:
            # detection map averaged over the 8 views of the square group in the (y, x) plane:
            # identity, 3 flips, 2 90-degree rotations, transpose and anti-transpose. Each view is
            # undone before aggregating. Stabilizes detections that flicker between frames; no
            # thresholds or weights involved.
            det_logits = [d.clone() for d in det_logits]
            views = [(lambda a: a.flip((-1,)), lambda a: a.flip((-1,))),
                     (lambda a: a.flip((-2,)), lambda a: a.flip((-2,))),
                     (lambda a: a.flip((-2, -1)), lambda a: a.flip((-2, -1))),
                     (lambda a: torch.rot90(a, 1, (-2, -1)), lambda a: torch.rot90(a, -1, (-2, -1))),
                     (lambda a: torch.rot90(a, 3, (-2, -1)), lambda a: torch.rot90(a, -3, (-2, -1))),
                     (lambda a: a.transpose(-1, -2), lambda a: a.transpose(-1, -2)),
                     (lambda a: a.flip((-2, -1)).transpose(-1, -2),
                      lambda a: a.flip((-2, -1)).transpose(-1, -2))]
            per_view = [[d.clone() for d in det_logits]]
            for fwd, inv in views:
                acc = [torch.zeros_like(d) for d in det_logits]
                for mm in models:
                    _, dv = mm.encode(fwd(imgs))
                    for f in range(len(det_logits)):
                        acc[f] = acc[f] + inv(dv[f]) / len(models)
                per_view.append(acc)
            if TTA_AGG == "max":
                det_logits = [torch.stack([v[f] for v in per_view]).amax(0) for f in range(len(det_logits))]
            elif TTA_AGG == "min":
                det_logits = [torch.stack([v[f] for v in per_view]).amin(0) for f in range(len(det_logits))]
            else:
                det_logits = [sum(v[f] for v in per_view) / len(per_view) for f in range(len(det_logits))]
        w = imgs.shape[1]
        for k in range(w):
            if s + k not in nodes_t:
                nodes_t[s + k] = find_peaks(det_logits[k][0, 0].float(), pk)
                if DUMP:
                    ix = nodes_t[s + k].long()
                    det_t[s + k] = det_logits[k][0, 0].float()[ix[:, 0], ix[:, 1], ix[:, 2]].cpu()

        def features(mi, kk, c):
            mm = models[mi]
            m = torch.ones(1, len(c), dtype=torch.bool, device=device)
            uf = mm._index_features(unet_outs[mi][:, kk], c[None], m)
            tc = torch.full((1, len(c), 1), float(kk), device=device)
            pe = pos_embed_fn(torch.cat([tc, c[None]], -1), pos_shape)
            phys = mm._index_features(SL.physical_maps(imgs, kk), c[None], m)
            return uf, torch.cat([pe, phys], -1), m

        def pair_logits(ka, ca, kb, cb):
            """Mean logits (edges, null row and daughter head) over all models."""
            tot = None
            for mi, mm in enumerate(models):
                (ua, pa, ma), (ub, pb, mb) = features(mi, ka, ca), features(mi, kb, cb)
                L = mm.predict_edges(ua, ub, ca[None] * ds_t, cb[None] * ds_t, pa, pb, ma, mb).float()
                tot = L if tot is None else tot + L
            return tot / len(models)

        for k in range(w - 1):
            t = s + k
            if t in done:
                continue
            done.add(t)
            c0, c1 = nodes_t[t], nodes_t[t + 1]
            if len(c0) == 0 or len(c1) == 0:
                continue
            L = pair_logits(k, c0, k + 1, c1)
            # with the daughter head the output has N_DAUGHTER_CLASSES extra columns: joint assignment
            daughters = (L[0, :len(c0), len(c1):len(c1) + SL.N_DAUGHTER_CLASSES]
                         if L.shape[2] > len(c1) else None)
            if DUMP and daughters is not None:
                for i_, ph in enumerate(torch.softmax(daughters.float(), -1).cpu().tolist()):
                    p_daughters[(t, i_)] = ph
            if DIVISIONS == "jaccard" and daughters is not None:
                pairs.append((t, L[0, :len(c0), :len(c1)].cpu(), L[0, -1, :len(c1)].cpu(), daughters.cpu()))
                continue
            for i, j, _p in SL.decode(L[0, :len(c0), :len(c1)], L[0, -1, :len(c1)], daughters):
                edges.append((t, i, t + 1, j))
                p_edge[(t, i, t + 1, j)] = _p
        if SKIPS and w >= 3:
            for k in range(w - 2):
                t = s + k
                if t in done_skip:
                    continue
                done_skip.add(t)
                c0, c2 = nodes_t[t], nodes_t[t + 2]
                if len(c0) == 0 or len(c2) == 0:
                    continue
                L2 = pair_logits(k, c0, k + 2, c2)
                skips.append((t, L2[0, :len(c0), :len(c2)].cpu(), L2[0, -1, :len(c2)].cpu()))

    if pairs:
        # DIVISIONS THAT MAXIMIZE THE EXPECTED JACCARD. With P(2 daughters) from the head over the
        # whole video: sort and take the k that maximize E[TP] / (E[annotated] + k - E[TP]), with
        # E[TP] = sum of the k largest and E[annotated] = sum of all. No threshold: it comes from the
        # model's own probabilities. The fraction of annotated cells cancels in the ratio.
        p2 = [torch.softmax(h.float(), -1)[:, 2] for _t, _L, _n, h in pairs]
        all_p = torch.cat(p2).double()
        order = torch.argsort(all_p, descending=True)
        c = torch.cumsum(all_p[order], 0)
        k = torch.arange(1, len(all_p) + 1, dtype=torch.float64)
        J = c / (all_p.sum() + k - c)
        k_best = int(torch.argmax(J)) + 1 if len(J) and float(J.max()) > 0 else 0
        chosen = torch.zeros(len(all_p), dtype=torch.bool)
        chosen[order[:k_best]] = True
        DIAG["divisions_chosen"] = k_best
        DIAG["expected_jaccard"] = float(J[k_best - 1]) if k_best else 0.0
        a = 0
        for (t, Lr, null, h), p in zip(pairs, p2):
            two = chosen[a:a + len(p)]
            a += len(p)
            for i, j, _p in SL.decode(Lr, null, h, two=two.numpy()):
                edges.append((t, i, t + 1, j))
                p_edge[(t, i, t + 1, j)] = _p

    if skips:
        # LEARNED GAP CLOSING: a track end at t and a track start at t+2 are joined if the model
        # itself (pair t, t+2) prefers it over "no parent"; optimal one-to-one assignment
        # (Hungarian) and a node at t+1 at the midpoint. No distances or thresholds: the model's
        # probability decides.
        has_child = {(a, i) for a, i, _b, _j in edges}
        has_parent = {(b, j) for _a, _i, b, j in edges}
        extra: dict[int, list] = {}
        n_closed = 0
        for t, L2, null in sorted(skips, key=lambda x: x[0]):
            ends = [i for i in range(L2.shape[0]) if (t, i) not in has_child]
            begins = [j for j in range(L2.shape[1]) if (t + 2, j) not in has_parent]
            if not ends or not begins:
                continue
            lp = torch.log_softmax(torch.cat([L2, null[None]], 0).double(), 0).numpy()
            cost = np.full((len(ends) + len(begins), len(begins)), 1e9)
            cost[:len(ends)] = -lp[ends][:, begins]
            cost[len(ends) + np.arange(len(begins)), np.arange(len(begins))] = -lp[-1, begins]
            ri, ci = SL.linear_sum_assignment(cost)
            for r_, c_ in zip(ri, ci):
                if r_ >= len(ends):
                    continue
                i, j = ends[r_], begins[c_]
                mid = (nodes_t[t][i] + nodes_t[t + 2][j]) / 2.0
                lst = extra.setdefault(t + 1, [])
                new_idx = len(nodes_t[t + 1]) + len(lst)
                lst.append(mid)
                edges.append((t, i, t + 1, new_idx))
                edges.append((t + 1, new_idx, t + 2, j))
                p_edge[(t, i, t + 1, new_idx)] = p_edge[(t + 1, new_idx, t + 2, j)] = float(np.exp(lp[i, j]))
                has_child.add((t, i)); has_parent.add((t + 2, j))
                has_child.add((t + 1, new_idx)); has_parent.add((t + 1, new_idx))
                n_closed += 1
        for t1, lst in extra.items():
            nodes_t[t1] = torch.cat([nodes_t[t1], torch.stack(lst)], 0)
        DIAG["gaps_closed"] = n_closed

    nodes, ident = [], {}
    for t, c in sorted(nodes_t.items()):
        full = (c * ds_t).cpu().numpy()
        for n, (z, y, x) in enumerate(full):
            ident[(t, n)] = len(nodes) + 1
            nodes.append({"node_id": len(nodes) + 1, "t": t, "z": float(z), "y": float(y), "x": float(x)})
    out_edges = [(ident[(t, i)], ident[(t1, j)]) for t, i, t1, j in edges]
    if DUMP:
        # features for track selection and division diagnostics; they do not change the output
        nan = float("nan")
        DUMPED.clear()
        DUMPED["det"] = [float(det_t[t][n]) if t in det_t and n < len(det_t[t]) else nan
                         for t, c in sorted(nodes_t.items()) for n in range(len(c))]
        DUMPED["daughters"] = [p_daughters.get((t, n), [nan, nan, nan])
                               for t, c in sorted(nodes_t.items()) for n in range(len(c))]
        DUMPED["p_edge"] = [p_edge.get(a, nan) for a in edges]
    return nodes, out_edges


def validate(model, videos: list, device, window_size: int, pool_kernel_um: float, pos_embed_fn,
             max_frames: int | None = None) -> dict:
    """Official score over `videos` (list of paths without extension, like the train ones)."""
    from tracking_cellmot.io import open_dataset
    from tracking_cellmot.metrics import evaluate, per_sample_metrics
    was = model.training
    model.eval()
    rows = []
    for vp in videos:
        vp = Path(vp)
        ds = open_dataset(vp, normalize=False, require_tracks=True, load_image=False,
                          downsample=tuple(DOWNSAMPLE))
        gt = ds.tracks
        if max_frames is not None:
            import tracksdata as td
            gt = gt.filter(td.NodeAttr("t") < max_frames).subgraph()
        nodes, out_edges = infer(model, ds, device, window_size, pool_kernel_um, pos_embed_fn, max_frames)
        er = evaluate(build_graph(nodes, out_edges), gt, scale=tuple(ds.scale), max_distance=SL.GATE_UM)
        census = census_of(vp)
        if max_frames is not None and census == census:
            census = census * min(1.0, max_frames / float(ds.image_shape[0]))
        m = per_sample_metrics(er, census, float("nan"))
        # per_sample_metrics does NOT return division_jaccard: compute it as the metric does, from
        # the crop's own division counts
        dd = er.division_tp + er.division_fp + er.division_fn
        dj = er.division_tp / dd if dd > 0 else float("nan")
        adj = m.get("adj_edge_jaccard", float("nan"))
        rows.append({"video": vp.name, "score": (adj if adj == adj else 0.0) + 0.1 * (dj if dj == dj else 0.0),
                     "J": m["edge_jaccard"], "J_adj": adj, "divJ": dj,
                     "nodes_over_census": len(nodes) / census if census and census == census else float("nan"),
                     "tp": er.edge_tp, "fp": er.edge_fp, "fn": er.edge_fn,
                     "div_tp": er.division_tp, "div_fp": er.division_fp, "div_fn": er.division_fn})
    model.train(was)
    return dict(aggregate_official(rows), per_video=rows)


def aggregate_official(rows: list) -> dict:
    """EXACT aggregation of the official metric (tracking_cellmot.metrics.summarise): J_adj
    weighted per crop with w = TP + FP + FN, MICRO edge and division Jaccard (summed counts),
    score = weighted J_adj + 0.1 x micro divJ. The plain per-crop mean used before over-weighted
    small crops (one with 70 edges and its single correct division added +0.2) and produced a false
    tie with the public chain (0.9051 vs 0.9028; with this aggregation 0.9002 vs ~0.925)."""
    if not rows:
        return {"score": float("nan"), "J": float("nan"), "J_adj": float("nan"), "divJ": float("nan"),
                "nodes_over_census": float("nan")}
    w = np.array([f["tp"] + f["fp"] + f["fn"] for f in rows], dtype=np.float64)
    adj = np.array([f["J_adj"] for f in rows], dtype=np.float64)
    ok = np.isfinite(adj) & (w > 0)
    j_adj = float((w[ok] * adj[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
    tp, fp, fn = (sum(f[k] for f in rows) for k in ("tp", "fp", "fn"))
    J = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    dtp, dfp, dfn = (sum(f.get(k, 0) or 0 for f in rows) for k in ("div_tp", "div_fp", "div_fn"))
    dj = dtp / (dtp + dfp + dfn) if dtp + dfp + dfn else float("nan")
    score = j_adj + (0.1 * dj if dj == dj else 0.0)
    return {"score": score, "J": J, "J_adj": j_adj, "divJ": dj,
            "nodes_over_census": float(np.nanmean([f["nodes_over_census"] for f in rows]))}


DOWNSAMPLE = [1, 4, 4]   # set by the training loop before calling validate()
# "map": each parent takes its most probable number of daughters. "jaccard": divisions are chosen
# over the whole video to maximize the expected division Jaccard.
DIVISIONS = "map"
DIAG: dict = {}
DUMP = False             # store per-node det / daughters and per-edge p in DUMPED
DUMPED: dict = {}
LAST_IDX: dict = {}
# SKIPS: (t, t+2) pair and learned gap closing. TTA: detection averaged over the 8 in-plane views.
SKIPS = False
TTA = False
