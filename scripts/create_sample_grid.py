#!/usr/bin/env python3
"""Build a true-colour preview grid directly from the EuroSAT ZIP archive."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path, PurePosixPath

import matplotlib
import numpy as np
from rasterio.io import MemoryFile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EXPECTED_CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

DISPLAY_NAMES = {
    "AnnualCrop": "Annual crop",
    "HerbaceousVegetation": "Herbaceous vegetation",
    "PermanentCrop": "Permanent crop",
    "SeaLake": "Sea / lake",
}


def contrast_stretch(composite: np.ndarray) -> np.ndarray:
    """Apply a shared 2nd-98th percentile stretch while preserving band ratios."""
    low, high = np.percentile(composite, (2, 98))
    if high <= low:
        return np.zeros_like(composite, dtype=np.float32)
    return np.clip((composite.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def select_members(archive: zipfile.ZipFile) -> dict[str, str]:
    """Select the first canonical class-prefixed TIFF for every class."""
    selected: dict[str, str] = {}
    for member in sorted(archive.namelist()):
        path = PurePosixPath(member)
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        class_name = path.parent.name
        if class_name not in EXPECTED_CLASSES:
            continue
        if not path.name.startswith(f"{class_name}_"):
            continue
        selected.setdefault(class_name, member)

    missing = set(EXPECTED_CLASSES) - selected.keys()
    if missing:
        raise ValueError(f"Archive is missing classes: {', '.join(sorted(missing))}")
    return selected


def read_false_colour(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    """Read Sentinel-2 B08/B04/B03 as a vegetation-emphasizing composite."""
    with MemoryFile(archive.read(member)) as memory_file:
        with memory_file.open() as dataset:
            if dataset.count != 13:
                raise ValueError(f"Expected 13 bands in {member}, found {dataset.count}")
            composite = np.moveaxis(dataset.read([8, 4, 3]), 0, -1)
    return contrast_stretch(composite)


def create_grid(archive_path: Path, output_path: Path) -> None:
    """Render one representative tile from each land-cover class."""
    with zipfile.ZipFile(archive_path) as archive:
        selected = select_members(archive)
        previews = {
            class_name: read_false_colour(archive, selected[class_name])
            for class_name in EXPECTED_CLASSES
        }

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 5, figsize=(15, 6.3))
    figure.patch.set_facecolor("#0d1117")

    for axis, class_name in zip(axes.flat, EXPECTED_CLASSES):
        axis.imshow(previews[class_name])
        axis.set_title(DISPLAY_NAMES.get(class_name, class_name), fontsize=11, pad=8)
        axis.axis("off")

    figure.suptitle(
        "EuroSAT land-cover classes — Sentinel-2 false-colour composites (B08/B04/B03)",
        fontsize=15,
        y=0.98,
    )
    figure.text(
        0.5,
        0.018,
        "One canonical tile per class; vegetation appears red and water appears dark blue",
        ha="center",
        color="#aab2bf",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.025, right=0.975, top=0.89, bottom=0.075, wspace=0.08, hspace=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to the EuroSAT ZIP archive")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/eurosat_sample_grid.png"),
        help="Destination PNG (default: assets/eurosat_sample_grid.png)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    create_grid(arguments.archive, arguments.output)
