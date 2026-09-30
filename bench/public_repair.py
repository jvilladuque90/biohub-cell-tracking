"""Post-ILP graph repair from the public 0.947 Kaggle notebook, used as part of the decoding
stage (one model, one pipeline).

Provenance
----------
`filter_output_graph` and its module-level constants come from public Kaggle notebooks for the
"Biohub - Cell Tracking During Development" competition by **evgendvorkin** (building on work by
**sjlee101**). That module is third-party code and is **not redistributed** in this repository.
To run the bench, save the notebook's repair code as a standalone `.py` file and point to it with
the environment variable `BIOHUB_PUBLIC_MODULE` (or the `--public-module` CLI flag of the bench
scripts, which sets that variable).

Repair chain, in the notebook's order (`filter_output_graph`):
  1. edge filter (consecutive frames, <= 14 um)
  2. motion relink: per-frame-pair Hungarian assignment over ALL nodes,
     cost = |target - (pos + 0.5*velocity)| + 0.05*dist - bonus*learned_prob,
     two passes (tight / relaxed radius); REPLACES the ILP edges (kept as a prior)
  3. single parent per node (highest prob)
  4. 1-frame gap closing (Hungarian end->start, synthetic node at the midpoint) and strict
     2-frame gap recovery
  5. post-link safe divisions (sister geometry / divergence)
  6. isolated-node pruning; tracks shorter than 6 dropped with adaptive rescue; linear
     position smoothing (window 2, weight 0.8)

Not reproduced (by design of the public code itself): the "deepcenter" veto (an extra detector
of the author, not in the public package) and intensity refinement of synthetic nodes (needs the
image; here `dataset=None` -> midpoint). Both fall back to accept / midpoint.

Parameters are the notebook's header `os.environ[...]` overrides, set BEFORE importing the
public module, which reads them at import time. They override any value already in the
environment.
"""
from __future__ import annotations

import importlib.util
import os
import sys

ENV_MODULE = "BIOHUB_PUBLIC_MODULE"

# ---- header overrides of the public notebook (only those governing the repair)
_ENV = {
    "BIOHUB_OUTPUT_FILTER_SHORT_TRACKS": "1",
    "BIOHUB_MOTION_RELINK_LEARNED_BONUS": "1.0",
    "BIOHUB_GAP_CLOSE_MAX_GAP": "2",
    "BIOHUB_GAP_CLOSE_UM": "5.0",   # deployed value (notebook cell 1)
    "BIOHUB_GAP_DENSITY_ADAPTIVE": "1",
    "BIOHUB_GAP_DENSITY_REFERENCE_UM": "6.5",
    "BIOHUB_GAP_DENSITY_GAIN": "0.040",
    "BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM": "0.125",
    "BIOHUB_GAP_DENSITY_NEIGHBORS": "3",
    "BIOHUB_OUTPUT_MIN_TRACK_LEN": "6",
    "BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS": "1",
    "BIOHUB_OUTPUT_GAP2_RECOVERY": "1",
    "BIOHUB_SAFE_DIV_MAX_UM": "9.0",
    "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "14.0",
    "BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU": "0.6",
    "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "10.0",
    "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.0076",
    "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.00375",
    "BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE": "1",
    "BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN": "4",
    "BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": "0.88",
    "BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM": "3.0",
    "BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC": "0.012",
    "BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS": "120",
    "BIOHUB_MOTION_RELINK_TIGHT_UM": "6.0",   # deployed value (notebook cell 3 default)
    # without the author's extra detector the veto does not exist and the public code accepts
    "BIOHUB_USE_DEEPCENTER_VETO": "0",
    "BIOHUB_DEEPCENTER_GAP_VETO": "0",
    "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0",
}

_MODULE = None


def load_public_module(path: str | None = None):
    """Set the notebook's environment overrides, then import the third-party module from
    `path` (default: $BIOHUB_PUBLIC_MODULE). Cached per process."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    path = path or os.environ.get(ENV_MODULE)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Public repair module not found ({path!r}). Save the public notebook's "
            f"filter_output_graph code as a .py file and set {ENV_MODULE} to its path.")
    for k, v in _ENV.items():
        os.environ[k] = v
    spec = importlib.util.spec_from_file_location("public_notebook_repair", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)   # reads os.environ at import time
    _MODULE = mod
    return mod


def repair(coords, ilp_edges, stats_out=None, module_path: str | None = None):
    """coords: (N,4) [t,z,y,x] in VOXELS for ALL detections; ilp_edges: list of (i, j, prob)
    with row indices selected by the ILP. As in the organizer's geff, only nodes incident to
    some selected edge are kept. Returns (nodes, edges): dicts {node_id,t,z,y,x} (z,y,x may be
    float: synthetic nodes and smoothing) and (a, b) node_id pairs."""
    pub = load_public_module(module_path)
    incident = set()
    for i, j, _ in ilp_edges:
        incident.add(int(i))
        incident.add(int(j))
    nodes_by_id = {}
    for i in sorted(incident):
        t, z, y, x = coords[i]
        nodes_by_id[i + 1] = {"node_id": i + 1, "t": int(t), "z": float(z), "y": float(y),
                              "x": float(x)}
    raw_edges = [{"source_id": int(i) + 1, "target_id": int(j) + 1, "edge_prob": float(p)}
                 for i, j, p in ilp_edges]
    nodes_by_id, edges, st = pub.filter_output_graph(nodes_by_id, raw_edges, dataset=None,
                                                     deepcenter_bundle=None)
    if stats_out is not None:
        stats_out.append(st)
    nodes = [{"node_id": int(k), "t": int(v["t"]), "z": float(v["z"]), "y": float(v["y"]),
              "x": float(v["x"])} for k, v in sorted(nodes_by_id.items())]
    out_edges = [(int(e["source_id"]), int(e["target_id"])) for e in edges]
    return nodes, out_edges
