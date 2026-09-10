# -*- coding: utf-8 -*-
"""Evaluate a VFRefine checkpoint on a VFRefine-1.2M test/val split.

    python scripts/evaluate.py \
        --model runs/stage3_refine \
        --split_dir data/vfrefine-1.2m/test \
        --mode fewshot --num_refs 3 --max_samples 1000 \
        --out runs/stage3_refine/eval_test.json

Reports IoU / MSE / CD / DTW / RR together with the parse-success rate and
mean generation latency, optionally broken down by seen / unseen characters
(``--seen_chars`` file) when the test split mixes both subsets.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vfrefine.data.dataset import load_pairs, load_glyph_image
from vfrefine.evaluation.metrics import aggregate, evaluate_sample
from vfrefine.inference.pipeline import VFRefinePipeline


def load_char_list(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    chars: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chars.add(line[0])
    return chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="VFRefine checkpoint directory")
    ap.add_argument("--split_dir", required=True, help="split with pairs.jsonl + images/")
    ap.add_argument("--mode", choices=["zeroshot", "fewshot"], default="fewshot")
    ap.add_argument("--num_refs", type=int, default=3, help="K reference templates")
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--seen_chars", default=None, help="file with seen characters")
    ap.add_argument("--out", default=None, help="output JSON report path")
    ap.add_argument("--dump_predictions", default=None,
                    help="optional directory to dump generated SVGs")
    args = ap.parse_args()

    pipe = VFRefinePipeline(args.model)
    records = load_pairs(os.path.join(args.split_dir, "pairs.jsonl"))
    if args.max_samples > 0:
        records = records[: args.max_samples]
    seen_chars = load_char_list(args.seen_chars)

    # reference pool per font (canonical glyphs, same style)
    by_font: Dict[str, List[Dict]] = {}
    for r in records:
        by_font.setdefault(r["font_id"], []).append(r)

    if args.dump_predictions:
        os.makedirs(args.dump_predictions, exist_ok=True)

    results: List[Dict] = []
    results_seen: List[Dict] = []
    results_unseen: List[Dict] = []
    latencies: List[float] = []
    n_failed = 0

    for rec in tqdm(records, desc="evaluating"):
        image = load_glyph_image(os.path.join(args.split_dir, rec["image"]))
        refs = []
        if args.mode == "fewshot":
            pool = [
                r for r in by_font[rec["font_id"]]
                if r["codepoint"] != rec["codepoint"]
            ][: args.num_refs]
            refs = [
                (load_glyph_image(os.path.join(args.split_dir, r["image"])), r["tgt_svg"])
                for r in pool
            ]

        t0 = time.perf_counter()
        try:
            pred_svg = pipe.refine(
                image, rec["src_svg"], references=refs,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - keep the eval loop alive
            print(f"[warn] generation failed for {rec['font_id']}/{rec['char']}: {exc}")
            n_failed += 1
            continue
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if args.dump_predictions:
            out_path = os.path.join(
                args.dump_predictions, f"{rec['font_id']}_{rec['codepoint']:04x}.svg"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(pred_svg)

        metrics = evaluate_sample(
            pred_svg, rec["tgt_svg"], src_svg=rec["src_svg"]
        )
        if metrics is None:
            n_failed += 1
            continue
        results.append(metrics)
        if seen_chars is not None:
            (results_seen if rec["char"] in seen_chars else results_unseen).append(metrics)

    report = {
        "model": args.model,
        "split": args.split_dir,
        "mode": args.mode,
        "num_refs": args.num_refs if args.mode == "fewshot" else 0,
        "n_total": len(records),
        "n_failed": n_failed,
        "latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
        "overall": aggregate(results),
    }
    if seen_chars is not None:
        report["seen_characters"] = aggregate(results_seen)
        report["unseen_characters"] = aggregate(results_unseen)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print("\n" + text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
