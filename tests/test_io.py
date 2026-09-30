"""I/O tests on tiny synthetic OME-Zarr v3 / GEFF stores (no competition data)."""
from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.storage import ZipStore

from cellmot import (
    SCALE_ZYX,
    list_zip_datasets,
    open_dataset,
    open_zip_dataset,
    open_zip_store,
)
from cellmot.io import nodes_microns

SHAPE = (3, 4, 8, 8)  # (T, Z, Y, X)
NODE_IDS = np.array([1, 2, 3, 4], dtype=np.uint64)
NODES = np.array([[0, 1, 2, 3], [1, 1, 2, 4], [2, 1, 2, 5], [2, 2, 6, 5]], dtype=np.int64)
EDGES = np.array([[1, 2], [2, 3], [2, 4]], dtype=np.uint64)


def _write_crop(store, split: str, dataset_id: str, with_geff: bool) -> np.ndarray:
    image = np.arange(np.prod(SHAPE), dtype=np.uint16).reshape(SHAPE)
    root = zarr.open_group(store=store, path=f"{split}/{dataset_id}.zarr", mode="a")
    root.create_array("0", data=image, chunks=(1, *SHAPE[1:]))
    if with_geff:
        g = zarr.open_group(store=store, path=f"{split}/{dataset_id}.geff", mode="a")
        g.create_array("nodes/ids", data=NODE_IDS)
        for i, k in enumerate(("t", "z", "y", "x")):
            g.create_array(f"nodes/props/{k}/values", data=NODES[:, i])
        g.create_array("edges/ids", data=EDGES)
    return image


def _check_tracks(ds):
    assert ds.has_tracks()
    np.testing.assert_array_equal(ds.node_ids, NODE_IDS)
    np.testing.assert_array_equal(ds.nodes, NODES)
    np.testing.assert_array_equal(ds.edges, EDGES)
    assert ds.nodes.dtype == np.int64 and ds.edges.dtype == np.uint64


@pytest.fixture
def zip_path(tmp_path):
    path = tmp_path / "mini.zip"
    store = ZipStore(str(path), mode="w")
    _write_crop(store, "train", "aaaa_0001", with_geff=True)
    _write_crop(store, "train", "bbbb_0002", with_geff=True)
    _write_crop(store, "test", "cccc_0003", with_geff=False)
    store.close()
    return path


def test_list_zip_datasets(zip_path):
    expected = {"train": ["aaaa_0001", "bbbb_0002"], "test": ["cccc_0003"]}
    assert list_zip_datasets(zip_path) == expected
    # A ZipStore opens lazily; listing only reads its `.path`, so it is never opened here.
    assert list_zip_datasets(open_zip_store(zip_path)) == expected


def test_open_zip_dataset(zip_path):
    store = open_zip_store(zip_path)
    try:
        ds = open_zip_dataset(store, "aaaa_0001", split="train")
        assert ds.image.shape == SHAPE
        assert ds.image.dtype == np.uint16
        assert ds.n_frames == SHAPE[0]
        frame = ds.frame(1)
        assert frame.shape == SHAPE[1:]
        np.testing.assert_array_equal(
            frame, np.arange(np.prod(SHAPE), dtype=np.uint16).reshape(SHAPE)[1])
        _check_tracks(ds)

        test_ds = open_zip_dataset(store, "cccc_0003", split="test")
        assert test_ds.image.shape == SHAPE
        assert not test_ds.has_tracks()
    finally:
        store.close()


def test_open_dataset_from_folder(tmp_path):
    store = zarr.storage.LocalStore(str(tmp_path))
    image = _write_crop(store, "train", "aaaa_0001", with_geff=True)
    ds = open_dataset(tmp_path, "aaaa_0001", split="train")
    assert ds.image.shape == SHAPE and ds.image.dtype == np.uint16
    np.testing.assert_array_equal(ds.frame(2), image[2])
    np.testing.assert_array_equal(ds.scale_zyx, SCALE_ZYX)
    _check_tracks(ds)


def test_nodes_microns():
    um = nodes_microns(NODES)
    assert um.shape == (len(NODES), 3) and um.dtype == np.float64
    np.testing.assert_allclose(um[0], [1 * 1.625, 2 * 0.40625, 3 * 0.40625])
    np.testing.assert_allclose(um[3], [2 * 1.625, 6 * 0.40625, 5 * 0.40625])
