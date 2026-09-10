# -*- coding: utf-8 -*-
"""Training utilities: backbone loading, stage-wise freezing, YAML configs."""

import json
import os
from typing import Dict, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

from ..models import VFRefineConfig, VFRefineForConditionalGeneration

logger = logging.get_logger(__name__)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(
    model_cfg: Dict,
    torch_dtype=torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> VFRefineForConditionalGeneration:
    """Instantiate VFRefine and load the Qwen2.5-Coder-7B backbone weights.

    The language-model weights (``model.*`` / ``lm_head.*``) are initialised
    from the specified checkpoint; the vision tower, multi-scale projectors
    and hierarchical heads are randomly initialised and learned from scratch.
    """
    overrides = model_cfg.get("config_overrides", {})
    config = VFRefineConfig(**overrides)
    # attention implementation must be selected before module construction
    config.text_config._attn_implementation = attn_implementation
    config.vision_config._attn_implementation = attn_implementation
    model = VFRefineForConditionalGeneration(config)
    model = model.to(torch_dtype)

    backbone = model_cfg.get("backbone")
    if backbone:
        logger.info(f"loading language backbone from {backbone}")
        qwen = AutoModelForCausalLM.from_pretrained(
            backbone, torch_dtype=torch_dtype
        )
        missing, unexpected = model.load_state_dict(qwen.state_dict(), strict=False)
        del qwen
        unexpected = [k for k in unexpected if not k.startswith("visual")]
        if unexpected:
            logger.warning(f"unexpected keys while loading backbone: {unexpected}")
        n_missing = len(missing)
        expected_missing_prefixes = ("visual.", "multi_scale_projectors.", "hier_head.")
        bad_missing = [k for k in missing if not k.startswith(expected_missing_prefixes)]
        if bad_missing:
            logger.warning(f"LM weights missing after backbone load: {bad_missing[:8]} ...")
        logger.info(f"backbone loaded ({n_missing} newly-initialised VFRefine parameters)")

    return model


def set_trainable(model: VFRefineForConditionalGeneration, stage: int) -> None:
    """Stage-wise freezing scheme (paper Sec. "Training Strategy").

    * Stage I (SSP): the visual encoder is frozen; the language model is
      fine-tuned on raw SVG text. (The vision tower receives no gradients at
      all because Stage-I batches contain no images.)
    * Stage II (R2V): the vision tower, projectors and LM are jointly trained
      for raster--vector alignment.
    * Stage III: full-parameter refinement training.
    """
    for p in model.parameters():
        p.requires_grad = True

    if stage == 1:
        for p in model.visual.parameters():
            p.requires_grad = False
        for p in model.multi_scale_projectors.parameters():
            p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"stage {stage}: {n_trainable / 1e9:.2f}B / {n_total / 1e9:.2f}B trainable parameters"
    )


def save_model(model, tokenizer, output_dir: str) -> None:
    """Save a VFRefine checkpoint (weights + config + tokenizer)."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    logger.info(f"checkpoint written to {output_dir}")
