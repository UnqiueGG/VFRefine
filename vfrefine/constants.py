# -*- coding: utf-8 -*-
"""Global constants for VFRefine.

Centralises the SVG command vocabulary, the glyph canvas resolution and the
chat-template fragments shared by data processing, training and inference.
"""

# ---------------------------------------------------------------------------
# SVG path command vocabulary (Paper Sec. 3.1, command set V = {M,L,H,V,C,Q,Z})
# ---------------------------------------------------------------------------
SVG_COMMANDS = ["M", "L", "H", "V", "C", "Q", "Z"]

# number of scalar parameters each command takes
#   M: x y        L: x y
#   H: x          V: y
#   C: x1 y1 x2 y2 x y
#   Q: x1 y1 x y
#   Z: (none)
SVG_COMMAND_ARITY = {
    "M": 2,
    "L": 2,
    "H": 1,
    "V": 1,
    "C": 6,
    "Q": 4,
    "Z": 0,
}

COMMAND_TO_IDX = {c: i for i, c in enumerate(SVG_COMMANDS)}
IDX_TO_COMMAND = {i: c for c, i in COMMAND_TO_IDX.items()}
NUM_COMMANDS = len(SVG_COMMANDS)

# ---------------------------------------------------------------------------
# Glyph canvas
# ---------------------------------------------------------------------------
GLYPH_RESOLUTION = 512          # raster glyphs are rendered at 512 x 512
VIEWBOX_SIZE = 512              # SVG viewBox="0 0 512 512"
MAX_COORD = VIEWBOX_SIZE - 1    # quantised integer coordinates in [0, 511]

# ---------------------------------------------------------------------------
# SVG scaffolding used to (de)serialise a glyph
# ---------------------------------------------------------------------------
SVG_HEADER = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
)
SVG_PATH_OPEN = '<path d="'
SVG_PATH_CLOSE = '"/></svg>'
SVG_FOOTER = SVG_PATH_CLOSE

# ---------------------------------------------------------------------------
# Chat template (Qwen2.5 style) and instruction templates
# (Paper Appendix, Tab. "Instruction templates")
# ---------------------------------------------------------------------------
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"

SYSTEM_FEWSHOT = (
    "You are a helpful assistant. Your task is to refine non-canonical initial "
    "SVG code into industrial-grade standard code with compact topology. Please "
    "refer to the target style exemplar and adopt an optimal control point "
    "distribution strategy to remove geometric redundancy while strictly "
    "preserving visual fidelity."
)

SYSTEM_ZEROSHOT = (
    "You are a helpful assistant. Your task is to refine non-canonical SVG code "
    "into compact code that meets industrial standards. Without external "
    "references, you independently infer the optimal topological structure of "
    "the input character. By accurately analyzing the visual features of the "
    "source image and removing the geometric redundancy of the original paths, "
    "you reconstruct an SVG sequence that is minimal, precise, and consistent "
    "with professional design standards."
)

SYSTEM_R2V = (
    "You are a helpful assistant. Your task is to convert the raster glyph image "
    "into standard SVG code. Analyze the visual structure of the character and "
    "produce compact, topologically clean vector contours."
)

USER_FEWSHOT = (
    "Reference images: {ref_imgs}; reference canonical SVG: {ref_svgs}; "
    "source image: {src_img}; source non-canonical SVG: {src_svg}"
)

USER_ZEROSHOT = "Source image: {src_img}; source non-canonical SVG: {src_svg}"

USER_R2V = "Source image: {src_img}"

# special token ids in the Qwen2.5 tokenizer (kept as fallbacks; the collator
# resolves them from the tokenizer whenever possible)
QWEN_IM_START_ID = 151644
QWEN_IM_END_ID = 151645
QWEN_ENDOFTEXT_ID = 151643
QWEN_VISION_START_ID = 151652
QWEN_VISION_END_ID = 151653
QWEN_IMAGE_PAD_ID = 151655
