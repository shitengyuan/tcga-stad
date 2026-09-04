"""Assemble a transparent, case-level M1 attention-visualization figure.

The source images were generated from the locked attention-overlap package.
They are whole-slide thumbnails, not pathologist annotations.  No morphology is
named or inferred here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image


CASES = [
    (
        "TCGA-FP-7998",
        "EBV-positive, M1-positive | M1 score 0.982",
        "TCGA-FP-7998/TCGA-FP-7998-01Z-00-DX1.9f24585a-d609-4094-8170-b0f483606a18_gt_original_vs_attention_overlap.jpg",
    ),
    (
        "TCGA-CG-4469",
        "CIN, M1-negative | M1 score 0.019",
        "TCGA-CG-4469/TCGA-CG-4469-01Z-00-DX1.005fafe8-ec9e-4283-95e4-f3cea0ac2365_gt_original_vs_attention_overlap.jpg",
    ),
]


def halves(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    midpoint = width // 2
    return image.crop((0, 0, midpoint, height)), image.crop((midpoint, 0, width, height))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), gridspec_kw={"wspace": 0.015, "hspace": 0.06})
    for row, (case_id, description, relative_path) in enumerate(CASES):
        image = Image.open(args.source_dir / relative_path).convert("RGB")
        original, overlay = halves(image)
        for col, panel in enumerate((original, overlay)):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_axis_off()
            if col == 0:
                ax.set_title(f"{'AB'[row]}. {case_id}: {description}", loc="left", fontsize=8.2,
                             fontweight="bold", pad=5)
    fig.text(0.25, 0.986, "Original H&E whole-slide thumbnail", ha="center", va="top", fontsize=10, fontweight="bold")
    fig.text(0.75, 0.986, "M1 attention overlay", ha="center", va="top", fontsize=10, fontweight="bold")
    mapper = ScalarMappable(norm=Normalize(0, 1), cmap="turbo")
    cax = fig.add_axes((0.40, 0.025, 0.20, 0.018))
    cb = fig.colorbar(mapper, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["lower relative attention", "higher relative attention"])
    cb.ax.tick_params(labelsize=7, length=0)
    fig.text(0.5, 0.063,
             "Attention is normalized within each slide (50th to 99.5th percentile); colors describe relative patch weights, not a pathology diagnosis or an absolute scale across cases.",
             ha="center", va="bottom", fontsize=7.1, color="#374151")
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.03, dpi=600)


if __name__ == "__main__":
    main()
