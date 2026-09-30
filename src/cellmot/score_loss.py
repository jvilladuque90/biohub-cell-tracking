"""CONSTANT-FREE MODEL: physical features, score-aligned loss, and the model's own decoder.

This module is copied into the organizer's repo (scripts/) and called by the hooks installed by
training/patch_score.py. Everything here is tested on CPU with synthetic data
(tests/test_score_loss.py).

WHY (audit of the earlier unified-model loss):
  1. the false-positive rule must be EXACTLY the metric's: an edge s->t counts if s matches an
     annotated cell WITH a successor, or t matches one WITH a predecessor (not "touches an
     annotated cell");
  2. matching must be the metric's: optimal one-to-one at <= 7 um (not greedy at 5 um);
  3. the count factor has no relu: the metric rewards under-emitting as much as it penalizes
     over-emitting;
  4. what is trained and what is shipped must be the same thing: the decoder is the model's own
     decision (each cell picks a parent or "no parent"), not a linker with hand-set constants;
     that way the soft J is the expectation of the J that is shipped;
  5. the division term (0.1 x division Jaccard) is part of the loss.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

N_PHYS = 8                  # physical features per cell (see physical_maps)
GATE_UM = 7.0               # matching distance of the official metric
MAX_DAUGHTERS = 2           # structural: a cell divides into at most two
N_DAUGHTER_CLASSES = MAX_DAUGHTERS + 1   # classes of the daughter head: 0, 1 or 2


# ---------------------------------------------------------------------------------------------
# 1. PHYSICAL FEATURES (new model input)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def physical_maps(imgs: torch.Tensor, i: int) -> torch.Tensor:
    """(B, W, Z, Y, X) normalized -> (B, N_PHYS, Z, Y, X) maps for frame i.

    Channels (in the downsampled space, ~1.6 um per voxel, isotropic):
      0 mean brightness in a 3^3 cube          (cell level)
      1 std in the 3^3 cube                    (texture / contrast)
      2 brightness 3^3 - brightness 5^3        ("peakiness": small, compact nuclei score high)
      3 brightness(i) - brightness(i-1), same place (temporal DERIVATIVE of intensity)
      4 1 if there is a previous frame in the window, else 0
      5-7 shift of the local intensity centroid between i-1 and i (a MOTION estimate that does
          not need to know where the cell came from)
    """
    f = imgs[:, i:i + 1].float()
    m3 = F.avg_pool3d(f, 3, 1, 1)
    m5 = F.avg_pool3d(f, 5, 1, 2)
    s3 = (F.avg_pool3d(f * f, 3, 1, 1) - m3 * m3).clamp_min(0).sqrt()
    B, _, Z, Y, X = f.shape
    out = torch.zeros(B, N_PHYS, Z, Y, X, device=f.device, dtype=torch.float32)
    out[:, 0:1], out[:, 1:2], out[:, 2:3] = m3, s3, m3 - m5
    if i > 0:
        fp = imgs[:, i - 1:i].float()
        out[:, 3:4] = m3 - F.avg_pool3d(fp, 3, 1, 1)
        out[:, 4] = 1.0
        out[:, 5:8] = _local_centroid(f) - _local_centroid(fp)
    return out


def _local_centroid(g: torch.Tensor) -> torch.Tensor:
    """Offset (voxels) of the intensity centroid in a 5^3 cube relative to each voxel."""
    B, _, Z, Y, X = g.shape
    den = F.avg_pool3d(g, 5, 1, 2) + 1e-6
    grid = torch.meshgrid(*[torch.arange(n, device=g.device, dtype=torch.float32) for n in (Z, Y, X)],
                          indexing="ij")
    out = []
    for r in grid:
        r = r[None, None]
        out.append(F.avg_pool3d(g * r, 5, 1, 2) / den - r)
    return torch.cat(out, 1)


# ---------------------------------------------------------------------------------------------
# 2. MATCHING AS THE METRIC DOES IT (optimal, one-to-one, <= 7 um)
# ---------------------------------------------------------------------------------------------
def match_detections(det_um: np.ndarray, gt_um: np.ndarray, gate: float = GATE_UM) -> np.ndarray:
    """For each detection, return the index of the matched GT cell or -1."""
    out = np.full(len(det_um), -1, np.int64)
    if len(det_um) == 0 or len(gt_um) == 0:
        return out
    d = np.linalg.norm(det_um[:, None, :] - gt_um[None, :, :], axis=-1)
    near = np.where((d <= gate).any(1))[0]
    if len(near) == 0:
        return out
    sub = d[near]
    cost = np.where(sub <= gate, sub, 1e6)
    ri, ci = linear_sum_assignment(cost)
    for a, b in zip(ri, ci):
        if cost[a, b] <= gate:
            out[near[a]] = b
    return out


# ---------------------------------------------------------------------------------------------
# 3. SCORE-ALIGNED LOSS
# ---------------------------------------------------------------------------------------------
def parent_probs(logits_real: torch.Tensor, logit_null: torch.Tensor) -> torch.Tensor:
    """(ns, nt) + (nt,) -> (ns, nt): P(the parent of cell j is cell i), with "no parent" as an
    alternative (its mass is discarded). This is the organizer's normalization (softmax per
    target) plus the null row."""
    full = torch.cat([logits_real, logit_null[None, :]], 0)
    return torch.softmax(full.float(), 0)[:-1]


def prob_two_daughters(e: torch.Tensor) -> torch.Tensor:
    """(ns, nt) -> (ns,): P(cell i ends up with >= 2 daughters), assuming independent targets."""
    p = e.clamp(0.0, 1.0 - 1e-6)
    log_none = torch.log1p(-p).sum(1)
    p0 = torch.exp(log_none)
    p1 = p0 * (p / (1 - p)).sum(1)
    return (1 - p0 - p1).clamp(0.0, 1.0)


def pair_terms(e, y, out_valid, in_valid, div_gt, matched, p2=None):
    """Soft terms of ONE frame pair of ONE sample.

    e (ns, nt)       parent probability (parent_probs)
    y (ns, nt)       1 if (i -> j) is a GT edge between matched detections
    out_valid (ns)   detection i matches an annotated cell WITH a successor
    in_valid (nt)    detection j matches an annotated cell WITH a predecessor
    div_gt (ns)      detection i matches an annotated cell that DIVIDES
    matched (ns)     detection i matches some annotated cell
    p2 (ns)          P(cell i has 2 daughters) from the daughter head; without it, it is derived
                     from e assuming independent targets (that version predicted 8487 divisions
                     against 19 annotated: two daughters picking the same parent is not a division)
    """
    valid = (out_valid[:, None] | in_valid[None, :]).float()
    tp = (e * y).sum()
    fp = (e * valid * (1 - y)).sum()
    if p2 is None:
        p2 = prob_two_daughters(e) if e.shape[1] > 1 else torch.zeros(e.shape[0], device=e.device)
    tpd = (p2 * div_gt.float()).sum()
    fpd = (p2 * (matched & ~div_gt).float()).sum()
    return tp, fp, tpd, fpd


def split_logits(logits: torch.Tensor):
    """(B, M_t + 1, M_t1 + N_DAUGHTER_CLASSES) -> edges (B, M_t + 1, M_t1) and daughter head
    (B, M_t, N_DAUGHTER_CLASSES).

    The last N_DAUGHTER_CLASSES columns are the "how many daughters does this cell have" logits
    (0, 1, 2); the null ("no parent") row is the last row. The organizer's code slices
    [:nt, :nt1], so the extra columns never reach it."""
    return logits[:, :, :-N_DAUGHTER_CLASSES], logits[:, :-1, -N_DAUGHTER_CLASSES:]


def accumulate_pair(acc: dict, logits: torch.Tensor, det_t: tuple, det_t1: tuple,
                    gt_c_t: torch.Tensor, gt_c_t1: torch.Tensor, gt_m_t: torch.Tensor,
                    gt_m_t1: torch.Tensor, trans: torch.Tensor, voxel) -> None:
    """Add the terms of one frame pair of the whole batch to the accumulator.

    logits (B, M_t + 1, M_t1 + N_DAUGHTER_CLASSES)  predict_edges output: null row LAST and the
                                                    daughter head in the last columns (see split_logits)
    det_t, det_t1               frame_det tuples: (coords, pos, mask, matches, feats)
    gt_c_* (B, G, 3), gt_m_*    annotated cells and their mask (downsampled voxels)
    trans (B, G_t, G_t1)        GT transitions
    voxel                       physical size of the downsampled voxel (um)
    """
    vs = np.asarray(voxel, dtype=np.float64)
    dev = logits.device
    logits, daughters = split_logits(logits)
    for b in range(logits.shape[0]):
        ns, nt = int(det_t[2][b].sum()), int(det_t1[2][b].sum())
        gt_n, gt_n1 = int(gt_m_t[b].sum()), int(gt_m_t1[b].sum())
        T = trans[b, :gt_n, :gt_n1].float()
        acc["ngt"] += float(T.sum())
        acc["nd"] += int((T.sum(1) >= 2).sum())
        if ns == 0 or nt == 0 or gt_n == 0 or gt_n1 == 0:
            continue
        m_t = match_detections(det_t[0][b, :ns].detach().cpu().double().numpy() * vs,
                               gt_c_t[b, :gt_n].detach().cpu().double().numpy() * vs)
        m_t1 = match_detections(det_t1[0][b, :nt].detach().cpu().double().numpy() * vs,
                                gt_c_t1[b, :gt_n1].detach().cpu().double().numpy() * vs)
        mt, mt1 = torch.as_tensor(m_t, device=dev), torch.as_tensor(m_t1, device=dev)
        ok_t, ok_t1 = mt >= 0, mt1 >= 0
        st, st1 = mt.clamp(min=0), mt1.clamp(min=0)
        n_out, n_in = T.sum(1), T.sum(0)
        y = T[st][:, st1] * (ok_t[:, None] & ok_t1[None, :]).float()
        out_v = ok_t & (n_out[st] > 0)
        in_v = ok_t1 & (n_in[st1] > 0)
        div = ok_t & (n_out[st] >= 2)
        e = parent_probs(logits[b, :ns, :nt], logits[b, -1, :nt])
        lk = torch.log_softmax(daughters[b, :ns].float(), -1)
        tp, fp, tpd, fpd = pair_terms(e, y, out_v, in_v, div, ok_t, p2=lk[:, 2].exp())
        acc["tp"] = acc["tp"] + tp
        acc["fp"] = acc["fp"] + fp
        acc["tpd"] = acc["tpd"] + tpd
        acc["fpd"] = acc["fpd"] + fpd
        # the daughter head is supervised on cells matched to an annotated cell: its number of
        # annotated successors (0, 1 or 2) in the next frame
        if bool(ok_t.any()):
            n_daughters = n_out[st].clamp(max=N_DAUGHTER_CLASSES - 1).long()
            acc["ce_daughters"] = acc["ce_daughters"] - lk[ok_t].gather(1, n_daughters[ok_t][:, None]).sum()
            acc["n_daughters"] += int(ok_t.sum())


class DivisionMemory:
    """Division Jaccard WITH MEMORY across batches.

    Why: with 19 annotated divisions over 19 crops, almost no batch contains one; without memory
    the soft division Jaccard is 0 in those batches and gives NO gradient (not even against false
    divisions), while the daughter head's cross-entropy pushes "2 daughters" to 0 everywhere: the
    head learned to never predict a division (divJ 0 in 17 of 19 crops). With memory, each batch
    adds its (differentiable) counts to the accumulated counts of previous batches (moving average,
    no gradient): the gradient is that of the real Jaccard over many divisions, in every batch."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.tpd = self.fpd = self.nd = 0.0

    def jaccard(self, tpd, fpd, nd):
        eps = 1e-6
        dJ = (self.tpd + tpd) / (self.nd + nd + self.fpd + fpd + eps)
        self.tpd = self.decay * (self.tpd + float(tpd))
        self.fpd = self.decay * (self.fpd + float(fpd))
        self.nd = self.decay * (self.nd + float(nd))
        return dJ


def soft_score(acc: dict, memory: "DivisionMemory | None" = None) -> dict:
    """acc: sums of tp, fp, ngt, tpd, fpd, nd, and a list of (soft_count, census) per frame.
    With `memory`, the division Jaccard is computed on the accumulated counts (see DivisionMemory)."""
    eps = 1e-6
    J = acc["tp"] / (acc["ngt"] + acc["fp"] + eps)
    if acc["counts"]:
        fac = torch.stack([1.0 - 0.1 * (n - c) / c for n, c in acc["counts"]]).mean()
        # squared log-ratio: ~((n-c)/c)^2 near the census but bounded far from it (10x too many
        # weighs 5.3, not 81): it cannot dominate the loss if the detector drifts far off
        dev = torch.stack([torch.log((n + 1.0) / (c + 1.0)) ** 2 for n, c in acc["counts"]]).mean()
    else:
        fac = torch.ones((), device=J.device)
        dev = torch.zeros((), device=J.device)
    if memory is not None:
        dJ = memory.jaccard(acc["tpd"], acc["fpd"], acc["nd"])
        if not torch.is_tensor(dJ):
            dJ = torch.tensor(float(dJ), device=J.device)
    elif acc["nd"] > 0:
        dJ = acc["tpd"] / (acc["nd"] + acc["fpd"] + eps)
    else:
        dJ = torch.zeros((), device=J.device)
    # The metric clips J_adj at 0; with a negative factor (far too many cells) the product J*factor
    # would reward LOWERING J. A 0.05 floor inside the product keeps the score increasing with J.
    # The push back to the useful regime comes from `count_dev` (a separate loss term).
    fac_c = fac.clamp(min=0.05)
    ce = acc["ce_daughters"] / max(1, acc["n_daughters"])
    return {"score": J * fac_c + 0.1 * dJ, "J": J, "factor": fac, "divJ": dJ,
            "count_dev": dev, "ce_daughters": ce}


def new_accumulator(device):
    z = torch.zeros((), device=device)
    return {"tp": z, "fp": z, "tpd": z, "fpd": z, "ngt": 0.0, "nd": 0, "counts": [],
            "ce_daughters": z, "n_daughters": 0}


def exclusivity(det_logits: torch.Tensor, gt_c: torch.Tensor, gt_m: torch.Tensor, pool_kernel: tuple,
                voxel, floor: float = -4.0) -> torch.Tensor:
    """ONE detection per annotated cell.

    Why: after adding the daughter head, half of the FPs were an annotated source linked to an
    UNMATCHED detection that, in 71.5 % of cases, lies < 7 um from the true daughter: two peaks on
    the same nucleus, and the parent keeps the one that does not count. Here, for every annotated
    cell in the frame, the expected number of peaks within GATE_UM (the metric's matching distance)
    must be 1: (n - 1)^2 penalizes both the duplicate peak and the cell with no detection.

    det_logits (B, 1, Z, Y, X); gt_c (B, G, 3) downsampled voxels; gt_m (B, G); voxel (um per voxel)."""
    pad = tuple(k // 2 for k in pool_kernel)
    with torch.no_grad():
        peak = (det_logits == F.max_pool3d(det_logits, pool_kernel, 1, pad)) & (det_logits > floor)
    vs = torch.as_tensor(voxel, dtype=torch.float32, device=det_logits.device)
    out = []
    for b in range(det_logits.shape[0]):
        g = gt_c[b][gt_m[b]].float()
        if len(g) == 0:
            continue
        idx = torch.nonzero(peak[b, 0])
        if len(idx) == 0:
            out.append(torch.ones((), device=det_logits.device))    # no detection: n = 0 for all
            continue
        s = torch.sigmoid(det_logits[b, 0][idx[:, 0], idx[:, 1], idx[:, 2]].float())
        near = (torch.cdist(idx.float() * vs, g * vs) <= GATE_UM).float()     # (P, G)
        n = (s[:, None] * near).sum(0)
        out.append(((n - 1.0) ** 2).mean())
    return torch.stack(out).mean() if out else torch.zeros((), device=det_logits.device)


def soft_count(det_logits: torch.Tensor, pool_kernel: tuple, floor: float = -4.0) -> torch.Tensor:
    """(B, 1, Z, Y, X) -> (B,): expected number of cells the decoder will emit (sum of sigmoids
    at the local maxima). Differentiable w.r.t. the logits."""
    pad = tuple(k // 2 for k in pool_kernel)
    with torch.no_grad():
        peak = (det_logits == F.max_pool3d(det_logits, pool_kernel, 1, pad)) & (det_logits > floor)
    return (torch.sigmoid(det_logits) * peak.float()).flatten(1).sum(1)


# ---------------------------------------------------------------------------------------------
# 4. CONSTANT-FREE DECODER
# ---------------------------------------------------------------------------------------------
def decode(logits_real: torch.Tensor, logit_null: torch.Tensor,
           logit_daughters: torch.Tensor | None = None, two=None):
    """Model decision for one frame pair, with no thresholds. Returns a list of (i, j, prob).

    With the daughter head (ns, 3): JOINT ASSIGNMENT. It finds the most probable graph under the
    model itself, P(parent of each cell) x P(number of daughters of each parent):
        min  sum_j -log P(parent_j | j)  +  sum_i [-log P(k_i = n_i) + log P(k_i = 0)]
    with n_i <= 2. Each parent offers two slots; slot m costs log P(k=m-1) - log P(k=m) (the cost
    of going from m-1 to m daughters). Each cell also has its own "no parent" slot. Solved exactly
    with the Hungarian algorithm. Two daughters can no longer share a parent unless the model
    predicts a division, and the true parent is not left without its daughter (53 % of the FPs
    were false divisions or stolen parents).

    Without the head: each cell picks independently; if a parent ends up with more than
    MAX_DAUGHTERS, the most probable ones are kept.

    `two` (ns,) bool: divisions already decided for the whole video (maximum expected Jaccard).
    Chosen parents may take a second daughter at no extra cost (the edges decide); the rest take
    at most one."""
    if logits_real.shape[0] == 0 or logits_real.shape[1] == 0:
        return []
    full = torch.cat([logits_real, logit_null[None, :]], 0)
    if logit_daughters is not None:
        return _joint_assignment(full, logit_daughters, two)
    pr = torch.softmax(full.float(), 0)
    arg = pr.argmax(0)
    ns = logits_real.shape[0]
    by_parent = {}
    for j, i in enumerate(arg.tolist()):
        if i < ns:
            by_parent.setdefault(i, []).append((float(pr[i, j]), j))
    out = []
    for i, children in by_parent.items():
        children.sort(reverse=True)
        out += [(i, j, p) for p, j in children[:MAX_DAUGHTERS]]
    return out


def _joint_assignment(full: torch.Tensor, logit_daughters: torch.Tensor, two=None):
    """full (ns + 1, nt) parent logits with the null row last; logit_daughters (ns, 3)."""
    ns, nt = full.shape[0] - 1, full.shape[1]
    lp = torch.log_softmax(full.float(), 0).cpu().double().numpy()              # (ns + 1, nt)
    pr = np.exp(lp)
    lk = torch.log_softmax(logit_daughters.float(), -1).cpu().double().numpy()  # (ns, 3)
    slot = np.stack([lk[:, m - 1] - lk[:, m] for m in range(1, MAX_DAUGHTERS + 1)], 1)  # (ns, 2)
    # slots are interchangeable for the Hungarian solver: the second can never be cheaper than the
    # first (otherwise a single daughter would be charged at the second slot's price). Exact when
    # the daughter distribution is concave, which is the normal case (1 daughter most likely).
    slot[:, 1] = np.maximum(slot[:, 1], slot[:, 0])
    if two is not None:
        two = np.asarray(two, dtype=bool)
        slot[:, 0] = np.where(two, 0.0, slot[:, 0])
        slot[:, 1] = np.where(two, 0.0, 1e6)
    big = 1e9
    # rows: parent slots (ns * MAX_DAUGHTERS) + one "no parent" slot per target cell (nt)
    cost = np.full((ns * MAX_DAUGHTERS + nt, nt), big)
    for m in range(MAX_DAUGHTERS):
        cost[m * ns:(m + 1) * ns] = -lp[:ns] + slot[:, m:m + 1]
    cost[ns * MAX_DAUGHTERS + np.arange(nt), np.arange(nt)] = -lp[ns]
    rows, cols = linear_sum_assignment(cost)
    out = []
    for r, j in zip(rows, cols):
        if r < ns * MAX_DAUGHTERS:
            i = r % ns
            out.append((int(i), int(j), float(pr[i, j])))
    return out


# ---------------------------------------------------------------------------------------------
# 5. STABILITY: weight EMA and loading into widened layers
# ---------------------------------------------------------------------------------------------
class WeightEMA:
    """Exponential moving average of the weights: smooths training spikes and usually gives the
    best model. Validation and checkpointing use these weights."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ms, mm = self.shadow.state_dict(), model.state_dict()
        for k, v in ms.items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(mm[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(mm[k])


def load_expanding(model, state: dict) -> dict:
    """Load a pretrained state into a model whose INPUT layers are wider: the new columns are
    zero-initialized, so the model starts out computing exactly the same function."""
    own = model.state_dict()
    expanded = []
    for k, v in list(state.items()):
        if k in own and own[k].shape != v.shape:
            w = own[k]
            if v.dim() == 2 and w.dim() == 2 and v.shape[0] == w.shape[0] and v.shape[1] < w.shape[1]:
                new = torch.zeros_like(w)
                new[:, :v.shape[1]] = v
                state[k] = new
                expanded.append(f"{k} {tuple(v.shape)}->{tuple(w.shape)}")
            else:
                raise ValueError(f"incompatible shape for {k}: {tuple(v.shape)} vs {tuple(w.shape)}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"expanded": expanded, "missing": list(missing), "unexpected": list(unexpected)}


def lr_warmup(step: int, warmup: int) -> float:
    """Learning-rate factor: linear warmup (avoids the start-up spike)."""
    return min(1.0, (step + 1) / max(1, warmup))


__all__ = ["N_PHYS", "GATE_UM", "MAX_DAUGHTERS", "N_DAUGHTER_CLASSES", "split_logits", "DivisionMemory",
           "physical_maps", "match_detections", "parent_probs", "prob_two_daughters", "pair_terms",
           "accumulate_pair", "soft_score", "new_accumulator", "exclusivity", "soft_count", "decode",
           "WeightEMA", "load_expanding", "lr_warmup", "math"]
