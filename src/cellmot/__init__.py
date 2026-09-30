"""cellmot: data access and local evaluation for the Biohub Cell Tracking competition.

The I/O helpers read OME-Zarr v3 images and GEFF ground-truth graphs, either
from an extracted folder or streamed directly from the competition zip.
Evaluation with the official metric lives in :mod:`cellmot.evaluate`.
"""
from .io import (
    SCALE_TZYX,
    SCALE_ZYX,
    Dataset,
    list_zip_datasets,
    open_dataset,
    open_zip_dataset,
    open_zip_store,
)

__all__ = [
    "SCALE_TZYX",
    "SCALE_ZYX",
    "Dataset",
    "open_dataset",
    "open_zip_store",
    "open_zip_dataset",
    "list_zip_datasets",
]
