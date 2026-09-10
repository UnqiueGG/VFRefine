# -*- coding: utf-8 -*-
"""Evaluation metrics for vector glyph refinement (paper Sec. "Evaluation
Metrics").

* **IoU / MSE** — raster fidelity: generated and ground-truth SVGs are
  rendered to high-resolution binary/grayscale images and compared.
* **CD** — Chamfer Distance between point sets sampled along the contours;
  coordinates are normalised to [0, 1] by the canvas size.
* **DTW** — Dynamic Time Warping between the ordered coordinate sequences,
  reported as the average per-step cost (pixel units).
* **RR** — Reduction Ratio of drawing commands between the non-canonical
  source and the optimised output.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np

from ..constants import GLYPH_RESOLUTION
from ..data.svg_utils import (
    SVGParsingError,
    count_commands,
    parse_svg,
    render_svg_to_array,
    sample_points,
)


# ---------------------------------------------------------------------------
# raster metrics
# ---------------------------------------------------------------------------
def binary_iou(pred_gray: np.ndarray, gt_gray: np.ndarray, thresh: int = 128) -> float:
    """IoU = Area(I_des ∩ I_gt) / Area(I_des ∪ I_gt)."""
    a = pred_gray < thresh
    b = gt_gray < thresh
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if not a.any() and not b.any() else 0.0
    return float(np.logical_and(a, b).sum() / union)


def raster_mse(pred_gray: np.ndarray, gt_gray: np.ndarray) -> float:
    """MSE = (1/HW) Σ (I_des(i,j) − I_gt(i,j))²  on [0,1]-scaled intensities."""
    a = pred_gray.astype(np.float32) / 255.0
    b = gt_gray.astype(np.float32) / 255.0
    return float(np.mean((a - b) ** 2))


# ---------------------------------------------------------------------------
# geometric metrics
# ---------------------------------------------------------------------------
def chamfer_distance(P: np.ndarray, Q: np.ndarray, size: int = GLYPH_RESOLUTION) -> float:
    """Symmetric Chamfer distance on normalised coordinates.

    CD(P_des, P_gt) = (1/|P_des|) Σ_x min_y ||x−y||₂ + (1/|P_gt|) Σ_y min_x ||y−x||₂
    """
    if len(P) == 0 or len(Q) == 0:
        return float("nan")
    P = P.astype(np.float32) / size
    Q = Q.astype(np.float32) / size

    def min_dists(A, B, chunk=4096):
        out = np.empty(len(A), dtype=np.float32)
        for i in range(0, len(A), chunk):
            d = np.linalg.norm(A[i : i + chunk, None, :] - B[None, :, :], axis=-1)
            out[i : i + chunk] = d.min(axis=1)
        return out

    return float(min_dists(P, Q).mean() + min_dists(Q, P).mean())


def dtw_distance(P: np.ndarray, Q: np.ndarray, window: Optional[int] = None) -> float:
    """DTW with average per-step cost: D(n,m) / |W*|.

    Dynamic programming over the accumulated-cost matrix; an optional
    Sakoe-Chiba band (``window``) bounds the warping range for efficiency.
    Distances are measured in pixel units.
    """
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return float("nan")
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))

    inf = float("inf")
    D = np.full((n + 1, m + 1), inf, dtype=np.float64)
    D[0, 0] = 0.0
    P64 = P.astype(np.float64)
    Q64 = Q.astype(np.float64)
    for i in range(1, n + 1):
        j0 = max(1, i - window)
        j1 = min(m, i + window)
        d = np.linalg.norm(P64[i - 1][None, :] - Q64[j0 - 1 : j1], axis=1)
        for off, j in enumerate(range(j0, j1 + 1)):
            D[i, j] = d[off] + min(D[i - 1, j - 1], D[i, j - 1], D[i - 1, j])

    # |W*|: backtrack along the optimal warping path
    i, j, steps = n, m, 0
    while i > 0 and j > 0:
        steps += 1
        move = int(np.argmin((D[i - 1, j - 1], D[i, j - 1], D[i - 1, j])))
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
    return float(D[n, m] / max(steps, 1))


def reduction_ratio(n_cmd_src: int, n_cmd_out: int) -> float:
    """RR = (N_src - N_des) / N_src x 100%."""
    if n_cmd_src <= 0:
        return float("nan")
    return (n_cmd_src - n_cmd_out) / n_cmd_src * 100.0


# ---------------------------------------------------------------------------
# per-sample driver
# ---------------------------------------------------------------------------
def evaluate_sample(
    pred_svg: str,
    tgt_svg: str,
    src_svg: Optional[str] = None,
    size: int = GLYPH_RESOLUTION,
    sample_step: float = 2.0,
) -> Optional[Dict[str, float]]:
    """Compute all metrics for one prediction; ``None`` if unparseable."""
    try:
        pred_prims = parse_svg(pred_svg)
        tgt_prims = parse_svg(tgt_svg)
    except SVGParsingError:
        return None

    try:
        pred_gray = render_svg_to_array(pred_svg, size=size)
        gt_gray = render_svg_to_array(tgt_svg, size=size)
    except Exception:
        return None

    P = sample_points(pred_prims, step=sample_step)
    Q = sample_points(tgt_prims, step=sample_step)

    out = {
        "iou": binary_iou(pred_gray, gt_gray),
        "mse": raster_mse(pred_gray, gt_gray),
        "cd": chamfer_distance(P, Q, size=size),
        "dtw": dtw_distance(P, Q),
        "n_cmd_pred": count_commands(pred_prims),
        "n_cmd_tgt": count_commands(tgt_prims),
    }
    if src_svg is not None:
        try:
            src_prims = parse_svg(src_svg)
            out["n_cmd_src"] = count_commands(src_prims)
            out["rr"] = reduction_ratio(out["n_cmd_src"], out["n_cmd_pred"])
        except SVGParsingError:
            pass
    return out


def aggregate(results: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean aggregation with NaN skipping + parse-failure accounting."""
    keys = ("iou", "mse", "cd", "dtw", "rr")
    agg: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in results if k in r and not math.isnan(r[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    agg["n_parsed"] = float(len(results))
    return agg
