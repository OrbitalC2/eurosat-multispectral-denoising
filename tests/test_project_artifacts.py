"""Fast integrity tests for the published notebook and experiment artifacts."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "eurosat_multispectral_denoising.ipynb"


def build_dncnn(num_channels: int, depth: int = 17, features: int = 64) -> nn.Module:
    layers: list[nn.Module] = [
        nn.Conv2d(num_channels, features, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    ]
    for _ in range(depth - 2):
        layers.extend(
            [
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ]
        )
    layers.append(nn.Conv2d(features, num_channels, kernel_size=3, padding=1))
    return nn.Sequential(*layers)


class ProjectArtifactTests(unittest.TestCase):
    def test_notebook_is_valid_and_code_cells_compile(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertNotIn("kaggle", notebook.get("metadata", {}))

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                with self.subTest(cell=index):
                    ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")

        serialized = json.dumps(notebook)
        self.assertNotIn("/kaggle/", serialized)
        self.assertNotIn("best_ModelB(2)", serialized)

    def test_supplied_weights_match_the_documented_models(self) -> None:
        cases = (
            ("best_model_a.pth", 13, 570_893),
            ("best_model_b.pth", 1, 557_057),
        )
        for filename, channels, parameter_count in cases:
            with self.subTest(model=filename):
                state = torch.load(
                    ROOT / "models" / filename,
                    map_location="cpu",
                    weights_only=True,
                )
                model = build_dncnn(channels)
                model.load_state_dict(state, strict=True)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameter_count)

    def test_published_images_are_readable(self) -> None:
        images = [ROOT / "assets" / "eurosat_sample_grid.png"]
        images.extend(sorted((ROOT / "assets" / "results").glob("*.png")))
        self.assertEqual(len(images), 6)
        for path in images:
            with self.subTest(image=path.name):
                with Image.open(path) as image:
                    image.verify()

    def test_evaluation_report_contains_full_noise_sweep(self) -> None:
        report = (ROOT / "results" / "evaluation_results.txt").read_text(encoding="utf-8")
        for sigma in range(5, 76, 5):
            self.assertRegex(report, rf"(?m)^\s*{sigma}\s+")
        self.assertIn("Paired t-test", report)


if __name__ == "__main__":
    unittest.main()
