"""Build the final submission notebook: the public 0.947 notebook with our window-5 model's edges.

The public pipeline keeps its detection (organizer model + second seed + TTA + DeepCenter), its global
linking (ILP), gap closing, pruning, HOCT veto and all its constants. Only one thing changes: inside the
pair loop of its predict script, the edge logits of its main model are replaced by those of our
window-5 model (kaggle/w5_edges.py). This avoids any detection-calibration mismatch and any window
mismatch (our model encodes its own 5-frame window around each pair).

The block below is injected into the notebook cell that launches prediction (anchored on
`predict_cmd = [`), after the pipeline's own patches:
  1. copy the organizer repo (tracking_cellmot package) and apply our seed, fp16 and score patches
     -> /kaggle/working/repo_w5;
  2. drop w5_edges.py into the pipeline's scripts; add the import and a hook right after the call
     to its main model's predict_edges;
  3. environment: BIOHUB_W5_REPO and BIOHUB_W5_WEIGHTS (weights dataset, e.g. biohub-w5-pesos).

The base public notebook is third-party and is not redistributed: pass it with --base-notebook and
--base-metadata.

Usage:
  python kaggle/build_submission_w5.py --base-notebook <public.ipynb> --base-metadata <kernel-metadata.json> \
      --owner <your-kaggle-username> --out-dir kaggle_biohub_submit_w5
"""
from __future__ import annotations

import argparse
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Anchor in the public notebook (must match its source exactly).
ANCLA = "predict_cmd = ["

# Old-string in the pipeline's scripts/predict_unet_transformer.py (must match exactly).
GANCHO_OLD = '''            edge_logits_pair = model.predict_edges(
                unet_feat_src, unet_feat_tgt,
                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                p_pos_src, p_pos_tgt,
                p_mask_src, p_mask_tgt,
            )  # (1, n_src, n_tgt)
'''
GANCHO_NEW = GANCHO_OLD + '''            # W5 EDGES: replace the edge logits of the pipeline's main model with our window-5 model
            edge_logits_pair = w5_edge_logits(
                id(zarr_arr), lambda _t: _load_frame(zarr_arr, _t, target_shape, downsample), t_src, T,
                target_shape, q_low, q_high, p_coords_src, p_coords_tgt, ds_arr_t, device,
            ).to(edge_logits_pair.dtype)
'''

# Old-string in the same predict script (must match exactly).
IMPORT_OLD = "from biohub_tracking.models import TemporalUNet3D\n"
IMPORT_NEW = IMPORT_OLD + "from w5_edges import w5_edge_logits\n"

# Our files shipped inside the notebook, keyed by their path in this repository. They are
# re-created under /kaggle/working/w5_src/ with the same layout; score_loss.py is also placed
# next to the training patches because patch_score.py copies it into the organizer repo.
SHIPPED = {
    "training/patch_seed.py": "training/patch_seed.py",
    "training/patch_fp16.py": "training/patch_fp16.py",
    "training/patch_score.py": "training/patch_score.py",
    "training/validate_no_constants.py": "training/validate_no_constants.py",
    "src/cellmot/score_loss.py": "src/cellmot/score_loss.py",
    "training/score_loss.py": "src/cellmot/score_loss.py",
}
PATCHES = ("training/patch_seed.py", "training/patch_fp16.py", "training/patch_score.py")

BLOQUE = r'''
# ==== W5 EDGES IN THE PUBLIC PIPELINE (kaggle/build_submission_w5.py) ====
import glob as _w_glob
import os as _w_os
import shutil as _w_shutil
import subprocess as _w_sp
import sys as _w_sys
from pathlib import Path as _w_Path

_w_srcs = [q for q in _w_glob.glob("/kaggle/input/**/kaggle-cell-tracking-competition-*/pyproject.toml", recursive=True)
           if not _w_os.path.exists(_w_os.path.join(_w_os.path.dirname(q), "scripts", "score_loss.py"))
           and _w_os.path.isdir(_w_os.path.join(_w_os.path.dirname(q), "src", "tracking_cellmot"))]
if not _w_srcs:
    raise RuntimeError("W5: organizer repo not found (dataset biohub-baseline-royerlab)")
_w_repo = "/kaggle/working/repo_w5"
_w_src = "/kaggle/working/w5_src"
if not _w_os.path.exists(_w_repo):
    _w_shutil.copytree(_w_os.path.dirname(_w_srcs[0]), _w_repo)
    for _w_n, _w_txt in %(files)s.items():
        _w_f = _w_os.path.join(_w_src, _w_n)
        _w_os.makedirs(_w_os.path.dirname(_w_f), exist_ok=True)
        open(_w_f, "w", encoding="utf-8").write(_w_txt)
    for _w_p in %(patches)s:
        _w_r = _w_sp.run([_w_sys.executable, _w_os.path.join(_w_src, _w_p), _w_repo])
        if _w_r.returncode != 0:
            raise RuntimeError("W5: " + _w_p + " failed")
_w_weights = sorted(q for q in _w_glob.glob("/kaggle/input/**/edge_predictor_best.pth", recursive=True) if %(filter)s in q)
if not _w_weights:
    raise FileNotFoundError("W5: weights not found (" + %(filter)s + ")")
_w_os.environ["BIOHUB_W5_REPO"] = _w_repo
_w_os.environ["BIOHUB_W5_WEIGHTS"] = _w_weights[0]
_w_ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
(REPO_DIR / "scripts" / "w5_edges.py").write_text(%(w5_edges)s)
_w_s = _w_ps.read_text()
for _w_old, _w_new in ((%(import_old)s, %(import_new)s), (%(hook_old)s, %(hook_new)s)):
    if _w_s.count(_w_old) != 1:
        raise RuntimeError(f"W5: anchor not unique ({_w_s.count(_w_old)}): {_w_old[:60]!r}")
    _w_s = _w_s.replace(_w_old, _w_new, 1)
compile(_w_s, str(_w_ps), "exec")
_w_ps.write_text(_w_s)
print("W5: our edge model hooked in | repo", _w_repo, "| weights", _w_weights[0], flush=True)
# ==== end ====

'''


def _read(rel):
    return io.open(os.path.join(REPO_ROOT, rel), encoding="utf-8").read()


def main():
    ap = argparse.ArgumentParser(description="Build the W5-edges submission notebook from the public 0.947 notebook.")
    ap.add_argument("--base-notebook", required=True, help="public 0.947 notebook (.ipynb), not redistributed")
    ap.add_argument("--base-metadata", required=True, help="kernel-metadata.json of the public notebook")
    ap.add_argument("--owner", default="<your-kaggle-username>", help="Kaggle username that owns the new kernel")
    ap.add_argument("--name", default="submit-cadena-w5", help="kernel name (prefixed with 'biohub-')")
    ap.add_argument("--weights-dataset", default="<your-kaggle-username>/biohub-w5-pesos",
                    help="Kaggle dataset with our window-5 weights (edge_predictor_best.pth + config.json)")
    ap.add_argument("--organizer-dataset", default="<your-kaggle-username>/biohub-baseline-royerlab",
                    help="Kaggle dataset with the organizer repo (tracking_cellmot)")
    ap.add_argument("--out-dir", default=None, help="output directory (default: kaggle_biohub_<name>)")
    a = ap.parse_args()

    nb = json.load(io.open(a.base_notebook, encoding="utf-8"))
    idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and ANCLA in "".join(c["source"])]
    assert len(idx) == 1, idx
    src = "".join(nb["cells"][idx[0]]["source"])
    assert src.count(ANCLA) == 1
    files = {dst: _read(rel) for dst, rel in SHIPPED.items()}
    # ascii() keeps the injected literals pure ASCII even if a shipped file has non-ASCII characters.
    bloque = BLOQUE % dict(files=ascii(files), patches=ascii(PATCHES),
                           filter=ascii(a.weights_dataset.split("/")[-1]),
                           w5_edges=ascii(_read("kaggle/w5_edges.py")),
                           import_old=ascii(IMPORT_OLD), import_new=ascii(IMPORT_NEW),
                           hook_old=ascii(GANCHO_OLD), hook_new=ascii(GANCHO_NEW))
    compile(bloque, "bloque", "exec")
    nb["cells"][idx[0]]["source"] = src.replace(ANCLA, bloque + ANCLA)
    compile("".join(nb["cells"][idx[0]]["source"]), "cell", "exec")
    assert "\n".join("".join(c["source"]) for c in nb["cells"]).isascii(), "notebook is not pure ASCII"

    name = "biohub-" + a.name
    d = a.out_dir or "kaggle_" + name.replace("-", "_")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, name.replace("-", "_") + ".ipynb")
    json.dump(nb, io.open(f, "w", encoding="utf-8"), ensure_ascii=False)
    m = json.load(io.open(a.base_metadata, encoding="utf-8"))
    m.update(id=a.owner + "/" + name, title=name, code_file=os.path.basename(f))
    for ds in (a.weights_dataset, a.organizer_dataset):
        if ds not in m["dataset_sources"]:
            m["dataset_sources"].append(ds)
    io.open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8").write(json.dumps(m, indent=1))
    print(f"{f} | cell {idx[0]} | datasets {m['dataset_sources']}")


if __name__ == "__main__":
    main()
