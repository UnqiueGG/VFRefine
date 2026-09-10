# -*- coding: utf-8 -*-
"""Single-glyph inference CLI.

Zero-shot universal refinement:
    python scripts/infer.py --model runs/stage3_refine \
        --image glyph.png --src_svg glyph_raw.svg --out glyph_refined.svg

Few-shot reference-guided refinement (K reference image+SVG pairs):
    python scripts/infer.py --model runs/stage3_refine \
        --image glyph.png --src_svg glyph_raw.svg --out glyph_refined.svg \
        --ref ref1.png ref1.svg --ref ref2.png ref2.svg --ref ref3.png ref3.svg

End-to-end image vectorization (Stage-II capability):
    python scripts/infer.py --model runs/stage3_refine --image glyph.png --r2v --out glyph.svg
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vfrefine.inference.pipeline import VFRefinePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="VFRefine checkpoint directory")
    ap.add_argument("--image", required=True, help="source raster glyph (PNG)")
    ap.add_argument("--src_svg", default=None, help="non-canonical source SVG")
    ap.add_argument("--ref", nargs=2, action="append", metavar=("IMG", "SVG"),
                    default=[], help="reference image+SVG pair (repeatable)")
    ap.add_argument("--r2v", action="store_true",
                    help="image-to-vector generation without a source SVG")
    ap.add_argument("--out", required=True, help="output SVG path")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    args = ap.parse_args()

    pipe = VFRefinePipeline(args.model)

    if args.r2v:
        svg = pipe.image_to_svg(args.image, max_new_tokens=args.max_new_tokens)
    else:
        if args.src_svg is None:
            ap.error("--src_svg is required unless --r2v is set")
        with open(args.src_svg, "r", encoding="utf-8") as f:
            src_svg = f.read()
        refs = []
        for img_path, svg_path in args.ref:
            with open(svg_path, "r", encoding="utf-8") as f:
                refs.append((img_path, f.read()))
        svg = pipe.refine(
            args.image, src_svg, references=refs, max_new_tokens=args.max_new_tokens
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"refined SVG written to {args.out}")


if __name__ == "__main__":
    main()
