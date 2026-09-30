"""OUR MODEL'S EDGES INSIDE THE PUBLIC PIPELINE ("best of both worlds").

The public pipeline detects with ITS ensemble (organizer model + second seed + TTA) and cleans the
graph with ITS post-processing; here only the edges of its main model are replaced by ours (window
5, physical features, "no parent" row; only the (1, n_src, n_tgt) matrix is returned).

Our model is loaded from ANOTHER copy of the organizer's repo (package tracking_cellmot, patched
with seed + fp16 + score) through importlib under its own module name, so it does not clash with the
pipeline's package (biohub_tracking). For each (t, t+1) pair of the pipeline we encode our own
window of W frames containing t and t+1 (the last one is cached: consecutive pairs reuse it) and
read the features at the SAME cells the pipeline detected.

Injected into the pipeline's predict script; configured through the environment:
  BIOHUB_W5_REPO     path of the patched organizer repo
  BIOHUB_W5_WEIGHTS  checkpoint path(s), ";"-separated for an ensemble (each with a config.json
                     next to it)
(These were BIOHUB_NUEVO_REPO / BIOHUB_NUEVO_PESOS in earlier private versions.)
"""
import importlib.util as _w5_iu
import json as _w5_json
import os as _w5_os
import sys as _w5_sys

import torch as _w5_torch

_STATE = {"models": None}


def _w5_load(device):
    repo = _w5_os.environ["BIOHUB_W5_REPO"]
    weights = _w5_os.environ["BIOHUB_W5_WEIGHTS"]
    for p in (_w5_os.path.join(repo, "src"), _w5_os.path.join(repo, "scripts")):
        if p not in _w5_sys.path:
            _w5_sys.path.append(p)          # APPENDED: the pipeline's own modules keep priority
    spec = _w5_iu.spec_from_file_location("w5_tut", _w5_os.path.join(repo, "scripts", "train_unet_transformer.py"))
    tut = _w5_iu.module_from_spec(spec)
    spec.loader.exec_module(tut)
    import score_loss as sl
    from tracking_cellmot.models import TemporalUNet3D
    # ENSEMBLE: several checkpoints separated by ";" (e.g. W5 + a second W3 seed), each with its window
    models = []
    for path in weights.split(";"):
        cfg = _w5_json.load(open(_w5_os.path.join(_w5_os.path.dirname(path), "config.json")))
        unet = TemporalUNet3D(in_channels=1, out_channels=cfg["unet_out_channels"], layers=cfg["unet_layers"])
        m = tut.UNetNodeTransformer(unet=unet, unet_out_channels=cfg["unet_out_channels"], pos_feat_dim=4 * tut._POS_EMBED_DIM)
        info = sl.load_expanding(m, dict(_w5_torch.load(path, map_location="cpu", weights_only=True)))
        assert all(k.startswith("transformer.div_mlp") for k in info["missing"]) and not info["unexpected"], info
        with _w5_torch.no_grad():
            if info["missing"]:
                m.transformer.div_mlp[-1].weight.zero_()
                m.transformer.div_mlp[-1].bias.zero_()
        m.to(device).eval()
        models.append((m, int(cfg["window_size"]), {"cache": None}))
        print(f"W5_EDGES: model {path} | window {cfg['window_size']} | missing {info['missing']}", flush=True)
    _STATE["models"] = (models, tut, sl)


@_w5_torch.no_grad()
def w5_detection_maps(key, load_frame, ws, n_frames, n_out, q_low, q_high, device):
    """THIRD DETECTION SEED: detection maps (logits) of our model for the n_out pipeline frames
    ws .. ws + n_out - 1, each a (1, 1, Z, Y, X) tensor on the SAME grid as the pipeline.
    Our W-frame window containing ws .. ws + n_out - 1 is encoded (cached per (key, start)) and the
    loaded models are averaged (W5 and, if present, the second seed)."""
    if _STATE["models"] is None:
        _w5_load(device)
    models, tut, sl = _STATE["models"]
    result = None
    for m, W, st in models:
        W = min(W, n_frames)
        s = max(0, min(ws - 1, n_frames - W))
        if s + W < ws + n_out:
            s = max(0, ws + n_out - W)
        k = ws - s
        cache = st.setdefault("det", {})
        if cache.get("start") != (key, s):
            imgs = _w5_torch.stack([load_frame(s + i) for i in range(W)]).float()
            imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)[None].to(device)
            _, det = m.encode(imgs)
            cache["start"] = (key, s)
            cache["det"] = [d.float() for d in det]
        det = cache["det"]
        maps = [det[k + i] for i in range(n_out)]
        result = maps if result is None else [a + b for a, b in zip(result, maps)]
    return [d / len(models) for d in result]


@_w5_torch.no_grad()
def w5_edge_logits(key, load_frame, t_src, n_frames, target_shape, q_low, q_high,
                   p_coords_src, p_coords_tgt, ds_arr_t, device):
    """Logits (1, n_src, n_tgt) of our model for the pipeline's (t_src, t_src + 1) pair.
    load_frame(t) -> downsampled (Z, Y, X) tensor (the same reader the pipeline uses)."""
    if _STATE["models"] is None:
        _w5_load(device)
    models, tut, sl = _STATE["models"]
    total = None
    for m, W, st in models:
        W = min(W, n_frames)
        s = max(0, min(t_src - 1, n_frames - W))       # t_src and t_src + 1 inside, one context frame before
        k = t_src - s
        if st["cache"] is None or st["cache"][0] != (key, s):
            imgs = _w5_torch.stack([load_frame(s + i) for i in range(W)]).float()
            imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)[None].to(device)
            unet_out, _ = m.encode(imgs)
            st["cache"] = ((key, s), imgs, unet_out)
        _, imgs, unet_out = st["cache"]
        pos_shape = (W,) + tuple(target_shape)
        feats = []
        for kk, c in ((k, p_coords_src.float()), (k + 1, p_coords_tgt.float())):
            msk = _w5_torch.ones(1, c.shape[1], dtype=_w5_torch.bool, device=device)
            uf = m._index_features(unet_out[:, kk], c, msk)
            tc = _w5_torch.full((1, c.shape[1], 1), float(kk), device=device)
            pe = tut._pos_embed_torch(_w5_torch.cat([tc, c], -1), pos_shape)
            phys = m._index_features(sl.physical_maps(imgs, kk), c, msk)
            feats.append((uf, _w5_torch.cat([pe, phys], -1), msk))
        (u0, p0, m0), (u1, p1, m1) = feats
        L = m.predict_edges(u0, u1, p_coords_src.float() * ds_arr_t, p_coords_tgt.float() * ds_arr_t, p0, p1, m0, m1)
        L = L[..., :p_coords_src.shape[1], :p_coords_tgt.shape[1]].float()
        total = L if total is None else total + L
    return total / len(models)
