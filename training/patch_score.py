"""CONSTANT-FREE MODEL: hooks into the organizer's `train_unet_transformer.py`.

Apply AFTER training/patch_seed.py and training/patch_fp16.py. It copies score_loss.py and
validate_no_constants.py into the organizer repo's scripts/ (the logic lives there and is tested
on CPU by tests/test_score_loss.py) and leaves only small hooks in the trainer, each an exact
old-string replacement (assert count) followed by compile():

  G0  imports of the two modules
  G1  transformer input widened by the N_PHYS physical features
  G2  "no parent" row (null_mlp), daughter head and division head in simple_node_transformer.py
  G3  per-frame census in VideoMeta / loading / batch (the metric's count factor)
  G4  physical features concatenated to each cell's input (training AND evaluation)
  G5  score-aligned loss accumulated pair by pair (optimal matching at 7 um); G5b: (t, t+2) pair
  G6  total loss = aux x (organizer loss) - weight x soft score; statistics
  G7  learning-rate warmup before each step and weight EMA after it
  G8  torch.cuda.synchronize only when a GPU is present (allows the CPU test)
  G9  load the pretrained state widening the input (identical start) with "no parent" off
  G10 epoch loop: OFFICIAL constant-free validation, best-by-validation and best-by-train
      checkpoints, plateau -> halve the learning rate, early stopping, history.json
  G11 CLI arguments and globals
  G12 read pre-downsampled data when the zarr provides it

New CLI flags of the patched trainer: --aux-weight, --score-weight, --ema, --warmup, --patience,
--plateau, --val-n, --val-frames, --count-weight, --div-weight, --div-memory, --excl-weight,
--skip-weight.

Sources: score_loss.py is taken from ../src/cellmot/score_loss.py relative to this file (repository
layout), falling back to this file's own directory; validate_no_constants.py sits next to this file.

NOTE (coupling with the other patches): several old-strings here are text INSERTED by
training/patch_seed.py and training/patch_fp16.py, not organizer code: the `reanudo desde` resume
print (G9), `globals()["_INIT_STATE"] = args.init` (G11), the `_SCALER` lines (G7, G11); the new
epoch loop also calls `_estado_modelo` and relies on `last_path`, both defined by patch_seed.py.
They must stay byte-identical to what those patches insert (the Spanish text is kept on purpose).

Note: the state-dict names `null_mlp`, `hijas_mlp` ("daughters") and `div_mlp` are kept as-is
because trained checkpoints are stored under those keys.

Usage: python training/patch_score.py <repo_dir>
"""
from __future__ import annotations

import io
import os
import shutil
import sys

TRAIN = "scripts/train_unet_transformer.py"
MODEL = "src/tracking_cellmot/models/simple_node_transformer.py"
HERE = os.path.dirname(os.path.abspath(__file__))

# Modules copied into <repo>/scripts/: destination name -> candidate source paths.
SOURCES = {
    "score_loss.py": (os.path.join(HERE, "..", "src", "cellmot", "score_loss.py"),
                      os.path.join(HERE, "score_loss.py")),
    "validate_no_constants.py": (os.path.join(HERE, "validate_no_constants.py"),),
}

# G2: null row + daughter head + division head in the model
P_MODEL = [
    ('            nn.Linear(hidden_dim // 2, 1),\n'
     '        )\n',
     '            nn.Linear(hidden_dim // 2, 1),\n'
     '        )\n'
     '        # NULL ROW: "this cell has no parent", from the target cell feature.\n'
     '        self.null_mlp = nn.Sequential(\n'
     '            nn.Linear(hidden_dim, hidden_dim // 2),\n'
     '            nn.GELU(),\n'
     '            nn.Linear(hidden_dim // 2, 1),\n'
     '        )\n'
     '        # DAUGHTER HEAD: "how many daughters does this cell have" (0, 1, 2), from the source\n'
     '        # feature AFTER attending to the targets (cross-attention). Name kept for checkpoints.\n'
     '        self.hijas_mlp = nn.Sequential(\n'
     '            nn.Linear(hidden_dim, hidden_dim // 2),\n'
     '            nn.GELU(),\n'
     '            nn.Linear(hidden_dim // 2, 3),\n'
     '        )\n'
     '        # DIVISION HEAD: extra evidence for "2 daughters" from the GEOMETRY of the two candidate\n'
     '        # daughters (the two strongest edges of the source): distances, sister distance,\n'
     '        # symmetry, angle, midpoint and the two scores. This is what the tree-based divisor\n'
     '        # used (divJ 0.12) and what the daughter head cannot see.\n'
     '        self.div_mlp = nn.Sequential(\n'
     '            nn.Linear(3 * hidden_dim + 8, hidden_dim // 2),\n'
     '            nn.GELU(),\n'
     '            nn.Linear(hidden_dim // 2, 1),\n'
     '        )\n'),
    ('        logits = torch.cat(chunks, dim=1)  # (B, N_t, N_t1)\n',
     '        logits = torch.cat(chunks, dim=1)  # (B, N_t, N_t1)\n'
     '        null_row = self.null_mlp(k).transpose(1, 2)  # (B, 1, N_t1)\n'
     '        logits = torch.cat([logits, null_row], dim=1)\n'
     '        daughters = self.hijas_mlp(q)  # (B, N_t, 3)\n'
     '        if logits.shape[2] >= 2:\n'
     '            _scale = torch.tensor([1.625, 0.40625, 0.40625], device=q.device, dtype=torch.float32)\n'
     '            _lg = logits[:, :-1].float()\n'
     '            if mask_t1 is not None:\n'
     '                _lg = _lg.masked_fill(~mask_t1[:, None, :], -1e4)\n'
     '            _v, _ix = _lg.topk(2, dim=-1)  # (B, N_t, 2)\n'
     '            _B, _Nt = _ix.shape[:2]\n'
     '            _bi = torch.arange(_B, device=q.device)[:, None, None]\n'
     '            _kj = k[_bi, _ix]  # (B, N_t, 2, H)\n'
     '            _cj = coords_t1[_bi, _ix].float() * _scale  # (B, N_t, 2, 3) in um\n'
     '            _ci = coords_t.float()[:, :, None, :] * _scale\n'
     '            _d = (_cj - _ci).norm(dim=-1)  # (B, N_t, 2)\n'
     '            _d12 = (_cj[:, :, 0] - _cj[:, :, 1]).norm(dim=-1)\n'
     '            _sym = (_d[..., 0] - _d[..., 1]).abs() / (_d.sum(-1) + 1e-6)\n'
     '            _cos = ((_cj[:, :, 0] - _ci[:, :, 0]) * (_cj[:, :, 1] - _ci[:, :, 0])).sum(-1) / (_d[..., 0] * _d[..., 1] + 1e-6)\n'
     '            _mid = ((_cj[:, :, 0] + _cj[:, :, 1]) / 2 - _ci[:, :, 0]).norm(dim=-1)\n'
     '            _geo = torch.stack([_d[..., 0] / 10, _d[..., 1] / 10, _d12 / 10, _sym, _cos, _mid / 10,\n'
     '                                _v[..., 0].clamp(-30, 30) / 10, _v[..., 1].clamp(-30, 30) / 10], -1)\n'
     '            _div = self.div_mlp(torch.cat([q.float(), _kj[:, :, 0].float(), _kj[:, :, 1].float(), _geo], -1))\n'
     '            daughters = daughters + torch.cat([torch.zeros_like(daughters[..., :2]), _div.to(daughters.dtype)], -1)\n'
     '        daughters = torch.cat([daughters, torch.zeros_like(daughters[:, :1])], dim=1)  # (B, N_t + 1, 3)\n'
     '        logits = torch.cat([logits, daughters.to(logits.dtype)], dim=2)  # (B, N_t + 1, N_t1 + 3)\n'),
]

APPEND = "            frame_det.append((det_c, det_p, det_m, matches, unet_feat))\n"

NEW_LOOP = '''    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    globals()["_LR0"] = lr
    globals()["_STEP"] = 0
    globals()["_EMA"] = SL.WeightEMA(model, _EMA_DECAY) if _EMA_DECAY > 0 else None
    globals()["_MEM_DIV"] = SL.DivisionMemory() if _DIV_MEMORY else None
    VSC.DOWNSAMPLE = list(downsample)
    # per-epoch validation decodes as it will be shipped: with gap closing whenever the window
    # allows it (it pays at decode time even if the (t, t+2) pair is not trained)
    VSC.SKIPS = bool(window_size >= 3)
    import json as _json
    _hist = []
    best_val, best_tr, no_improve = -1e9, -1e9, 0
    save_path = output_dir / "edge_predictor_best.pth"
    best_tr_path = output_dir / "edge_predictor_best_train.pth"
    last_path = output_dir / "edge_predictor_last.pth"
    (output_dir / "config_score.json").write_text(_json.dumps({
        "window_size": window_size, "downsample": list(downsample), "pool_kernel_um": pool_kernel_um,
        "n_phys": SL.N_PHYS, "null_row": True, "aux": _AUX_W, "score_weight": _SCORE_W, "ema": _EMA_DECAY,
        "warmup": _WARMUP, "patience": _PATIENCE, "plateau": _PLATEAU, "div_weight": _DIV_W,
        "div_memory": _DIV_MEMORY, "excl_weight": _EXCL_W, "skip_weight": _SKIP_W}))
    val_videos = list(test_files)[:_VAL_N] if _VAL_N else list(test_files)
    print(f"OFFICIAL constant-free validation on {len(val_videos)} videos: "
          f"{[Path(v).name for v in val_videos]} (frames: {_VAL_FRAMES or 'all'})", flush=True)
    print(f"Loss = {_AUX_W} x organizer - {_SCORE_W} x soft score - {_DIV_W} x extra divJ | EMA {_EMA_DECAY} | "
          f"warmup {_WARMUP} steps | plateau {_PLATEAU} | patience {_PATIENCE}", flush=True)
    # starting point: the loaded model before training (separates what the decoder brings from
    # what training brings)
    t0 = time.monotonic()
    val = VSC.validate(model, val_videos, device, window_size, pool_kernel_um, _pos_embed_torch,
                       max_frames=_VAL_FRAMES)
    print(f"  EPOCH  -1 (untrained) | OFFICIAL VALIDATION score {val['score']:.4f} J {val['J']:.4f} "
          f"J_adj {val['J_adj']:.4f} divJ {val['divJ']:.4f} nodes/census {val['nodes_over_census']:.2f} | "
          f"{time.monotonic() - t0:.0f}s", flush=True)
    for epoch in range(n_epochs):
        globals()["_STATS"] = {}
        t0 = time.monotonic()
        edge_loss, det_loss = train_epoch(
            model, train_loader, optimizer, device, det_loss_weight, det_neg_weight,
            max_iters=max_iters, pool_kernel_um=pool_kernel_um,
        )
        st = {k: float(np.mean(v)) for k, v in globals()["_STATS"].items() if v}
        t_tr = time.monotonic() - t0
        evaluated = globals()["_EMA"].shadow if globals()["_EMA"] is not None else model
        t0 = time.monotonic()
        val = VSC.validate(evaluated, val_videos, device, window_size, pool_kernel_um,
                           _pos_embed_torch, max_frames=_VAL_FRAMES)
        t_val = time.monotonic() - t0
        state = _estado_modelo(evaluated)
        torch.save(state, last_path)
        improved = val["score"] > best_val + 1e-4
        if improved:
            best_val, no_improve = val["score"], 0
            torch.save(state, save_path)
        else:
            no_improve += 1
            if _PLATEAU and no_improve % _PLATEAU == 0:
                globals()["_LR0"] *= 0.5
                print(f"  plateau: {no_improve} epochs without improvement -> lr {globals()['_LR0']:.2e}", flush=True)
        if st.get("score", -1e9) > best_tr:
            best_tr = st["score"]
            torch.save(state, best_tr_path)
        row = {"epoch": epoch, "lr": globals()["_LR0"], "org_edge": edge_loss, "org_det": det_loss,
               "train_score": st.get("score"), "train_J": st.get("J"), "train_factor": st.get("factor"),
               "train_divJ": st.get("divJ"), "train_count_dev": st.get("count_dev"),
               "train_ce_daughters": st.get("ce_daughters"), "val_score": val["score"], "val_J": val["J"],
               "val_J_adj": val["J_adj"], "val_divJ": val["divJ"],
               "val_nodes_over_census": val["nodes_over_census"], "best_val": best_val,
               "no_improve": no_improve, "t_train_s": round(t_tr), "t_val_s": round(t_val),
               "per_video": val["per_video"]}
        _hist.append(row)
        (output_dir / "history.json").write_text(_json.dumps(_hist, indent=1))
        (output_dir / "last.json").write_text(_json.dumps({"epoch": epoch + 1, "score": val["score"]}))
        nan = float("nan")
        print(f"  EPOCH {epoch:3d} | TRAIN soft_score {st.get('score', nan):.4f} J {st.get('J', nan):.4f} "
              f"factor {st.get('factor', nan):.4f} divJ {st.get('divJ', nan):.4f} count {st.get('count_dev', nan):.3f} "
              f"daughters {st.get('ce_daughters', nan):.3f} excl {st.get('exclusivity', nan):.3f} "
              f"skipJ {st.get('skip_J', nan):.4f} | org edge {edge_loss:.5f} "
              f"det {det_loss:.5f} | OFFICIAL VALIDATION score {val['score']:.4f} J {val['J']:.4f} "
              f"J_adj {val['J_adj']:.4f} divJ {val['divJ']:.4f} nodes/census {val['nodes_over_census']:.2f} | "
              f"best {best_val:.4f}{' *' if improved else '  '} no improvement {no_improve} | "
              f"lr {globals()['_LR0']:.2e} | {t_tr:.0f}s + {t_val:.0f}s", flush=True)
        if _PATIENCE and no_improve >= _PATIENCE:
            print(f"EARLY STOP: {no_improve} epochs without validation improvement (best {best_val:.4f})",
                  flush=True)
            break
    print(f"\\nBest official constant-free validation: {best_val:.4f} -> {save_path}", flush=True)
    return model
'''


def _source(name):
    for cand in SOURCES[name]:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"{name} not found in {SOURCES[name]}")


def main(repo):
    for name in SOURCES:
        shutil.copy(_source(name), os.path.join(repo, "scripts", name))

    # G2 null row in the model
    pm = os.path.join(repo, MODEL)
    s = io.open(pm, encoding="utf-8").read()
    for old, new in P_MODEL:
        assert s.count(old) == 1, ("model", s.count(old), old[:50])
        s = s.replace(old, new, 1)
    compile(s, pm, "exec")
    io.open(pm, "w", encoding="utf-8").write(s)

    p = os.path.join(repo, TRAIN)
    s = io.open(p, encoding="utf-8").read()

    def one(old, new, n=1):
        nonlocal s
        c = s.count(old)
        assert c == n, (c, n, old[:70])
        s = s.replace(old, new)

    # G0
    one("from itertools import cycle as _cycle\n",
        "from itertools import cycle as _cycle\nimport score_loss as SL\nimport validate_no_constants as VSC\n")
    # G1
    one("            feat_dim=unet_out_channels + pos_feat_dim,\n",
        "            feat_dim=unet_out_channels + pos_feat_dim + SL.N_PHYS,\n")
    # G3 census
    one("    q_high: float                 # 99.9% quantile for normalization\n",
        "    q_high: float                 # 99.9% quantile for normalization\n"
        "    census_frame: float = 0.0     # estimated_number_of_nodes / T (count factor)\n")
    one('        q_high=float(ds.quantiles["0.999"]),\n    )\n',
        '        q_high=float(ds.quantiles["0.999"]),\n'
        '        census_frame=(lambda _c: _c / float(image_shape[0]) if _c == _c else 0.0)'
        '(VSC.census_of(Path(ds_path))),\n    )\n')
    one('            "downsample": torch.tensor(vm.downsample, dtype=torch.float32),\n        }\n',
        '            "downsample": torch.tensor(vm.downsample, dtype=torch.float32),\n'
        '            "census_frame": torch.tensor(float(vm.census_frame), dtype=torch.float32),\n        }\n')
    # G4 physical features (train_epoch and evaluate)
    one(APPEND,
        "            det_p = torch.cat([det_p, model._index_features(SL.physical_maps(imgs, i), det_c, det_m)], -1)\n"
        + APPEND, n=2)
    # G5 per-pair accumulator (train_epoch only)
    one("        block_losses = []\n        for i in range(W - 1):\n",
        "        block_losses = []\n        _acc = SL.new_accumulator(device)\n        _bad = False\n"
        "        for i in range(W - 1):\n")
    one("            edge_logits = edge_logits.float()\n",
        "            edge_logits = edge_logits.float()\n"
        "            # a batch with non-finite output (fp16 overflow) must not kill the run: the\n"
        "            # organizer's BCE aborts on CUDA. Sanitize it, zero the step further down\n"
        "            # (loss x 0), and count it.\n"
        "            if not bool(torch.isfinite(edge_logits).all()):\n"
        "                _bad = True\n"
        "                edge_logits = torch.nan_to_num(edge_logits.detach(), nan=0.0, posinf=0.0, neginf=0.0)\n"
        "            SL.accumulate_pair(_acc, edge_logits, frame_det[i], frame_det[i + 1], coords[:, i],\n"
        "                               coords[:, i + 1], masks[:, i], masks[:, i + 1], targets[:, i], voxel_size)\n")
    # G5b SKIP: the (t, t+2) pair inside a window of >= 3. The two-frame GT transitions are the
    # composition of the two consecutive ones. Same model, same loss (the organizer's + soft J on
    # that pair): the model learns to join a track with its continuation when the middle
    # detection is missing.
    one("        edge_loss = sum(block_losses) / len(block_losses)\n",
        "        _skip_J = torch.zeros((), device=device)\n"
        "        if _SKIP_W > 0 and W >= 3:\n"
        "            _acc2 = SL.new_accumulator(device)\n"
        "            for _i in range(W - 2):\n"
        "                _T2 = torch.bmm(targets[:, _i].float(), targets[:, _i + 1].float()).clamp(max=1.0)\n"
        "                _ns, _nt = frame_det[_i][0].shape[1], frame_det[_i + 2][0].shape[1]\n"
        "                _pt2 = build_matched_edge_targets(frame_det[_i][3], frame_det[_i + 2][3], _T2, _ns, _nt)\n"
        "                with torch.autocast(\"cuda\", dtype=torch.float16, enabled=_AMP):\n"
        "                    _L2 = model.predict_edges(\n"
        "                        frame_det[_i][4], frame_det[_i + 2][4],\n"
        "                        frame_det[_i][0] * ds_scale, frame_det[_i + 2][0] * ds_scale,\n"
        "                        frame_det[_i][1], frame_det[_i + 2][1],\n"
        "                        frame_det[_i][2], frame_det[_i + 2][2],\n"
        "                    )\n"
        "                _L2 = _L2.float()\n"
        "                if not bool(torch.isfinite(_L2).all()):\n"
        "                    _bad = True\n"
        "                    _L2 = torch.nan_to_num(_L2.detach(), nan=0.0, posinf=0.0, neginf=0.0)\n"
        "                SL.accumulate_pair(_acc2, _L2, frame_det[_i], frame_det[_i + 2], coords[:, _i],\n"
        "                                   coords[:, _i + 2], masks[:, _i], masks[:, _i + 2], _T2, voxel_size)\n"
        "                block_losses.append(compute_batch_loss(_L2, _pt2, frame_det[_i][2], frame_det[_i + 2][2]))\n"
        "            _skip_J = SL.soft_score(_acc2)[\"J\"]\n"
        "        edge_loss = sum(block_losses) / len(block_losses)\n")
    # G12 PRE-DOWNSAMPLED DATA: if the zarr has a "ds" array (pixels already at (1, 4, 4)), read it;
    # the original "0" array may then be empty and only provides the shape. Without "ds" nothing changes.
    one('        z = zarr.open_group(str(vm.zarr_path), mode="r")["0"]\n',
        '        _zg = zarr.open_group(str(vm.zarr_path), mode="r")\n'
        '        z = _zg["0"]\n')
    one('        raw = z[t_start : t_start + W, ::dz, ::dy, ::dx].astype(np.float32)\n',
        '        if "ds" in _zg:\n'
        '            raw = _zg["ds"][t_start : t_start + W].astype(np.float32)\n'
        '        else:\n'
        '            raw = z[t_start : t_start + W, ::dz, ::dy, ::dx].astype(np.float32)\n')
    # G6 total loss
    one("        loss = edge_loss + det_loss_weight * det_loss\n",
        "        _pk = VSC.kernel_nms(pool_kernel_um, voxel_size)\n"
        "        _cf = batch.get(\"census_frame\")\n"
        "        if _cf is not None:\n"
        "            for _i in range(W):\n"
        "                _n = SL.soft_count(det_logits[_i], _pk)\n"
        "                for _b in range(B):\n"
        "                    if float(_cf[_b]) > 0:\n"
        "                        _acc[\"counts\"].append((_n[_b], float(_cf[_b])))\n"
        "        _r = SL.soft_score(_acc, globals().get(\"_MEM_DIV\"))\n"
        "        _r[\"exclusivity\"] = (torch.stack([SL.exclusivity(det_logits[_i], coords[:, _i], masks[:, _i],\n"
        "                                  _pk, voxel_size) for _i in range(W)]).mean()\n"
        "                                  if _EXCL_W > 0 else torch.zeros((), device=device))\n"
        "        loss = (_AUX_W * (edge_loss + det_loss_weight * det_loss) - _SCORE_W * _r[\"score\"]\n"
        "                + _COUNT_W * _r[\"count_dev\"] - _DIV_W * _r[\"divJ\"]\n"
        "                + _AUX_W * _r[\"ce_daughters\"] + _EXCL_W * _r[\"exclusivity\"] - _SKIP_W * _skip_J)\n"
        "        _r[\"skip_J\"] = _skip_J\n"
        "        _ST = globals().setdefault(\"_STATS\", {})\n"
        "        _ST.setdefault(\"skipped_batches\", []).append(float(_bad or not bool(torch.isfinite(loss))))\n"
        "        if _bad or not bool(torch.isfinite(loss)):\n"
        "            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0) * 0.0\n"
        "        for _k in (\"score\", \"J\", \"factor\", \"divJ\", \"count_dev\", \"ce_daughters\", \"exclusivity\", \"skip_J\"):\n"
        "            _ST.setdefault(_k, []).append(float(_r[_k].detach()))\n")
    # G7 warmup and weight EMA
    one("        optimizer.zero_grad()\n        _SCALER.scale(loss).backward()\n",
        "        _f = SL.lr_warmup(globals().get(\"_STEP\", 0), _WARMUP)\n"
        "        for _g in optimizer.param_groups:\n"
        "            _g[\"lr\"] = globals().get(\"_LR0\", _g[\"lr\"]) * _f\n"
        "        optimizer.zero_grad()\n        _SCALER.scale(loss).backward()\n")
    one("        _SCALER.update()\n",
        "        _SCALER.update()\n"
        "        globals()[\"_STEP\"] = globals().get(\"_STEP\", 0) + 1\n"
        "        if globals().get(\"_EMA\") is not None:\n"
        "            globals()[\"_EMA\"].update(model)\n")
    # G8
    n = s.count("        torch.cuda.synchronize()\n")
    assert n >= 3, n
    s = s.replace("        torch.cuda.synchronize()\n",
                  "        if torch.cuda.is_available():\n            torch.cuda.synchronize()\n")
    # G9 widening load + null row switched off at start. The old-string is the text inserted by
    # training/patch_seed.py and must stay byte-identical to it.
    one("        _mis, _un = model.load_state_dict(_st, strict=False)\n"
        "        print(f\"  reanudo desde {_init}: {len(_mis)} missing, {len(_un)} unexpected\", flush=True)\n",
        "        _info = SL.load_expanding(model, dict(_st))\n"
        "        print(f\"  resumed from {_init}: expanded {_info['expanded']} | missing {_info['missing']} | \"\n"
        "              f\"unexpected {_info['unexpected']}\", flush=True)\n"
        "    else:\n"
        "        _info = {\"missing\": [k for k in model.state_dict()]}\n"
        "    with torch.no_grad():\n"
        "        # only the NEW heads (absent from the loaded state) are initialized: \"no parent\" off\n"
        "        # and a uniform daughter head (0, 1 and 2 daughters equally likely: the joint\n"
        "        # assignment starts with no preference and only enforces <= 2 daughters per parent)\n"
        "        if any(k.startswith(\"transformer.null_mlp\") for k in _info[\"missing\"]):\n"
        "            model.transformer.null_mlp[-1].weight.zero_()\n"
        "            model.transformer.null_mlp[-1].bias.fill_(_NULL_INIT)\n"
        "        if any(k.startswith(\"transformer.hijas_mlp\") for k in _info[\"missing\"]):\n"
        "            model.transformer.hijas_mlp[-1].weight.zero_()\n"
        "            model.transformer.hijas_mlp[-1].bias.zero_()\n"
        "        if any(k.startswith(\"transformer.div_mlp\") for k in _info[\"missing\"]):\n"
        "            # division head at zero: the model starts out doing the same as the loaded one\n"
        "            model.transformer.div_mlp[-1].weight.zero_()\n"
        "            model.transformer.div_mlp[-1].bias.zero_()\n")
    # G10 epoch loop (NEW_LOOP calls _estado_modelo, defined by training/patch_seed.py)
    i = s.index("    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)\n")
    j = s.index("    return model\n", i) + len("    return model\n")
    s = s[:i] + NEW_LOOP + s[j:]
    # G11 default globals and CLI arguments
    one('_SCALER = torch.amp.GradScaler("cuda", enabled=False)\n',
        '_SCALER = torch.amp.GradScaler("cuda", enabled=False)\n'
        '_AUX_W, _SCORE_W, _EMA_DECAY, _WARMUP = 1.0, 1.0, 0.999, 200\n'
        '_PATIENCE, _PLATEAU, _VAL_N, _VAL_FRAMES, _NULL_INIT, _COUNT_W = 5, 2, 4, None, -8.0, 0.1\n'
        '_DIV_W = 0.0   # EXTRA weight of the division Jaccard (the metric\'s 0.1 is already in the score)\n'
        '_DIV_MEMORY = False   # division Jaccard with memory across batches\n'
        '_EXCL_W = 0.0   # exclusivity weight: one detection per annotated cell\n'
        '_SKIP_W = 0.0   # weight of the (t, t+2) pair: learned linking over gaps\n')
    one('    parser.add_argument("--epochs", type=int, default=50)\n',
        '    parser.add_argument("--epochs", type=int, default=50)\n'
        '    parser.add_argument("--aux-weight", type=float, default=1.0)\n'
        '    parser.add_argument("--score-weight", type=float, default=1.0)\n'
        '    parser.add_argument("--ema", type=float, default=0.999)\n'
        '    parser.add_argument("--warmup", type=int, default=200)\n'
        '    parser.add_argument("--patience", type=int, default=5)\n'
        '    parser.add_argument("--plateau", type=int, default=2)\n'
        '    parser.add_argument("--val-n", type=int, default=4)\n'
        '    parser.add_argument("--val-frames", type=int, default=None)\n'
        '    parser.add_argument("--count-weight", type=float, default=0.1)\n'
        '    parser.add_argument("--div-weight", type=float, default=0.0)\n'
        '    parser.add_argument("--div-memory", action="store_true")\n'
        '    parser.add_argument("--excl-weight", type=float, default=0.0)\n'
        '    parser.add_argument("--skip-weight", type=float, default=0.0)\n')
    one('    globals()["_INIT_STATE"] = args.init\n',
        '    globals()["_INIT_STATE"] = args.init\n'
        '    globals().update(_AUX_W=args.aux_weight, _SCORE_W=args.score_weight, _EMA_DECAY=args.ema,\n'
        '                     _WARMUP=args.warmup, _PATIENCE=args.patience, _PLATEAU=args.plateau,\n'
        '                     _VAL_N=args.val_n, _VAL_FRAMES=args.val_frames,\n'
        '                     _COUNT_W=args.count_weight, _DIV_W=args.div_weight, _DIV_MEMORY=args.div_memory,\n'
        '                     _EXCL_W=args.excl_weight, _SKIP_W=args.skip_weight)\n')
    compile(s, p, "exec")
    io.open(p, "w", encoding="utf-8").write(s)
    print("score patch applied to", repo)


if __name__ == "__main__":
    main(sys.argv[1])
