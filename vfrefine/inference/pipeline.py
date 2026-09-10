# -*- coding: utf-8 -*-
"""VFRefine inference pipeline: zero-shot / few-shot refinement and
end-to-end image-to-vector generation."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from transformers import AutoTokenizer

from ..constants import GLYPH_RESOLUTION
from ..data.collator import VFRefineCollator
from ..data.dataset import GlyphImageProcessor, load_glyph_image
from ..models import VFRefineForConditionalGeneration


class VFRefinePipeline:
    """High-level inference wrapper.

    Args:
        model_path: directory of a saved VFRefine checkpoint (weights +
            ``config.json`` + tokenizer), or ``None`` to build from an
            explicitly passed ``model``.
        device / torch_dtype: placement.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[VFRefineForConditionalGeneration] = None,
        tokenizer_path: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        torch_dtype=torch.bfloat16,
    ):
        if model is None:
            if model_path is None:
                raise ValueError("provide either model_path or a model instance")
            model = VFRefineForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch_dtype
            )
        self.model = model.to(device).eval()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        vc = self.model.config.vision_config
        self.collator = VFRefineCollator(
            self.tokenizer,
            image_processor=GlyphImageProcessor(
                patch_size=vc.patch_size,
                temporal_patch_size=vc.temporal_patch_size,
                spatial_merge_size=vc.spatial_merge_size,
            ),
            spatial_merge_size=vc.spatial_merge_size,
        )

    # ------------------------------------------------------------------
    def _prepare_inputs(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Prompt + images -> model tensors (batch size 1)."""
        prompt, _ = self.collator.build_prompt(sample)
        ids = self.collator._encode(prompt)

        images: List[Image.Image] = []
        if sample["task"] == "r2v":
            images = [sample["image"]]
        elif sample["task"] == "refine":
            if sample.get("fewshot"):
                images.extend(r["image"] for r in sample["references"])
            images.append(sample["image"])

        pixel_values, grids = [], []
        for img in images:
            proc = self.collator.image_processor(img)
            pixel_values.append(proc["pixel_values"])
            grids.append(proc["image_grid_thw"])
        ids = self.collator.expand_image_placeholders(ids, grids)

        # generation starts right after `<path d="`: append the fixed scaffold
        scaffold = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" '
            'viewBox="0 0 512 512"><path d="'
        )
        ids += self.collator._encode(scaffold)

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0).to(self.device)
            batch["image_grid_thw"] = torch.stack(grids, dim=0).to(self.device)
        return batch

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, sample: Dict, max_new_tokens: int = 4096) -> str:
        """Run structured command--parameter decoding; returns the SVG text."""
        batch = self._prepare_inputs(sample)
        gen_ids = self.model.structured_generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            image_token_id=self.collator.image_pad_id,
            eos_token_ids=(self.collator.im_end_id, self.tokenizer.eos_token_id),
            max_new_tokens=max_new_tokens,
            command_token_ids=self.collator.command_token_ids,
            first_command_token_ids=self.collator.first_command_token_ids,
            d_close_token_id=self.collator.d_close_token_id,
            digit_token_ids=self.collator.digit_token_ids,
        )
        text = self.tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
        scaffold = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" '
            'viewBox="0 0 512 512"><path d="'
        )
        svg = scaffold + text
        # cut anything after the closing tag of the document
        end = svg.find("</svg>")
        if end >= 0:
            svg = svg[: end + len("</svg>")]
        return svg

    # ------------------------------------------------------------------
    def refine(
        self,
        image: Union[str, Image.Image],
        src_svg: str,
        references: Optional[Sequence[Tuple[Union[str, Image.Image], str]]] = None,
        max_new_tokens: int = 4096,
    ) -> str:
        """Vector glyph refinement.

        Args:
            image: source raster glyph (path or PIL image).
            src_svg: non-canonical SVG code of the source glyph.
            references: optional list of (reference image, canonical SVG)
                pairs; ``None`` / empty selects zero-shot mode.
        """
        if isinstance(image, str):
            image = load_glyph_image(image, size=GLYPH_RESOLUTION)
        refs = []
        for rimg, rsvg in references or []:
            if isinstance(rimg, str):
                rimg = load_glyph_image(rimg, size=GLYPH_RESOLUTION)
            refs.append({"image": rimg, "svg": rsvg})
        sample = {
            "task": "refine",
            "fewshot": bool(refs),
            "image": image,
            "src_svg": src_svg,
            "references": refs,
        }
        return self.generate(sample, max_new_tokens=max_new_tokens)

    def image_to_svg(
        self,
        image: Union[str, Image.Image],
        max_new_tokens: int = 4096,
    ) -> str:
        """End-to-end raster-to-vector conversion (Stage-II capability)."""
        if isinstance(image, str):
            image = load_glyph_image(image, size=GLYPH_RESOLUTION)
        return self.generate({"task": "r2v", "image": image}, max_new_tokens=max_new_tokens)
