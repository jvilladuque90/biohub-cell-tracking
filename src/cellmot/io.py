"""I/O for the Biohub Cell Tracking competition data.

Loads OME-Zarr v3 images (T, Z, Y, X, uint16) and GEFF v1.1 ground-truth graphs,
streaming chunks (one frame = one chunk, ~4.5 MB) so the ~80 GB archive never has
to be extracted.

Conventions:
  - Image axes: (T, Z, Y, X), single channel.
  - Physical voxel size (µm): SCALE = (t=1.0, z=1.625, y=0.40625, x=0.40625).
  - GEFF nodes: int64 props (t, z, y, x) in PIXEL coordinates (not microns).
  - Edges are directed node(t) -> node(t+1). A division is a node with two
    outgoing edges.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np

# Physical voxel size (µm); identical for every crop.
SCALE_TZYX = (1.0, 1.625, 0.40625, 0.40625)
SCALE_ZYX = np.asarray(SCALE_TZYX[1:], dtype=np.float64)  # for distances in µm


@dataclasses.dataclass
class Dataset:
    """One crop: a lazy image plus (optionally) its ground-truth graph."""
    dataset_id: str
    image: "object"                        # lazy zarr array, shape (T, Z, Y, X)
    scale_zyx: np.ndarray                  # µm per voxel (z, y, x)
    nodes: Optional[np.ndarray] = None     # (N, 4) int, columns [t, z, y, x]
    node_ids: Optional[np.ndarray] = None  # (N,) uint64
    edges: Optional[np.ndarray] = None     # (E, 2) uint64, [source_id, target_id]

    @property
    def n_frames(self) -> int:
        return int(self.image.shape[0])

    def frame(self, t: int) -> np.ndarray:
        """Load a single 3D frame (Z, Y, X) into memory (one zarr chunk)."""
        return np.asarray(self.image[t])

    def has_tracks(self) -> bool:
        return self.nodes is not None and self.edges is not None


def _open_zarr_image(root: Path | str):
    """Open the level-0 image array of an OME-Zarr v3 store."""
    import zarr
    store = zarr.open(str(root), mode="r")
    return store["0"]  # multiscale level "0"


def _read_geff_group(g) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read (node_ids, nodes[t,z,y,x], edges) from an open GEFF zarr group.

    GEFF layout:
      nodes/ids                     (N,)   uint64
      nodes/props/{t,z,y,x}/values  (N,)   int64
      edges/ids                     (E, 2) uint64  [source, target]
    """
    node_ids = np.asarray(g["nodes/ids"][:])
    cols = [np.asarray(g[f"nodes/props/{k}/values"][:]) for k in ("t", "z", "y", "x")]
    nodes = np.stack(cols, axis=1).astype(np.int64)
    edges = np.asarray(g["edges/ids"][:]).astype(np.uint64)
    return node_ids, nodes, edges


# --------------------------------------------------------------------------- #
# Reading straight from the competition zip (zarr ZipStore, no extraction)
# --------------------------------------------------------------------------- #
def open_zip_store(zip_path: Path | str):
    """Open the whole competition .zip as a read-only zarr store.

    The archive adds essentially no compression on top of the blosc chunks, so
    reading an individual member is cheap and random-access. Reuse the returned
    store for all reads so the zip index is parsed only once.
    """
    from zarr.storage import ZipStore
    return ZipStore(str(zip_path), mode="r")


def list_zip_datasets(store_or_path) -> dict:
    """Return ``{split: [dataset_id, ...]}`` by scanning the zip index.

    Accepts a ZipStore (its ``.path`` is used) or the path to the .zip.
    """
    import re
    import zipfile
    path = getattr(store_or_path, "path", store_or_path)
    ids = {"train": set(), "test": set()}
    pat = re.compile(r"^(train|test)/([^/]+)\.zarr/")
    with zipfile.ZipFile(str(path)) as zf:
        for key in zf.namelist():
            m = pat.match(key)
            if m:
                ids[m.group(1)].add(m.group(2))
    return {k: sorted(v) for k, v in ids.items()}


def open_zip_dataset(store, dataset_id: str, split: str = "train",
                     with_tracks: bool = True) -> Dataset:
    """Open one crop directly from a ZipStore, without extracting anything.

    Example:
        store = open_zip_store("biohub-...zip")
        ds = open_zip_dataset(store, "44b6_0113de3b", split="train")
        f0 = ds.frame(0)          # (64, 256, 256) uint16
        ds.nodes, ds.edges        # ground truth when split == "train"
    """
    import zarr
    image = zarr.open_array(store=store, path=f"{split}/{dataset_id}.zarr/0",
                            mode="r")
    ds = Dataset(dataset_id=dataset_id, image=image, scale_zyx=SCALE_ZYX.copy())
    if with_tracks and split == "train":
        g = zarr.open_group(store=store, path=f"{split}/{dataset_id}.geff",
                            mode="r")
        ds.node_ids, ds.nodes, ds.edges = _read_geff_group(g)
    return ds


def load_geff(geff_path: Path | str):
    """Read nodes and edges of a .geff directly from its zarr store.

    Returns ``(node_ids, nodes, edges)`` with ``nodes`` as (N, 4) [t, z, y, x]
    pixel coordinates and ``edges`` as (E, 2) [source_id, target_id].
    """
    import zarr
    return _read_geff_group(zarr.open(str(geff_path), mode="r"))


def open_dataset(root: Path | str, dataset_id: str, split: str = "train",
                 with_tracks: bool = True) -> Dataset:
    """Open a crop from an extracted folder containing ``{split}/{id}.zarr`` (+ ``.geff``).

    Example:
        ds = open_dataset("./data", "44b6_0113de3b", split="train")
        f0 = ds.frame(0)               # (64, 256, 256) uint16
        ds.nodes, ds.edges             # ground truth when split == "train"
    """
    root = Path(root)
    image = _open_zarr_image(root / split / f"{dataset_id}.zarr")
    ds = Dataset(dataset_id=dataset_id, image=image, scale_zyx=SCALE_ZYX.copy())
    if with_tracks and split == "train":
        ds.node_ids, ds.nodes, ds.edges = load_geff(root / split / f"{dataset_id}.geff")
    return ds


def nodes_microns(nodes_tzyx: np.ndarray) -> np.ndarray:
    """Convert [t, z, y, x] pixel columns to physical (z, y, x) positions in µm."""
    return nodes_tzyx[:, 1:].astype(np.float64) * SCALE_ZYX[None, :]
