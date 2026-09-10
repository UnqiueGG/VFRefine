# -*- coding: utf-8 -*-
"""VFRefine trainer: HF Trainer with hierarchical-loss logging and the
stroke-complexity curriculum sampler for Stage III."""

from typing import Dict, Optional

import torch
from transformers import Trainer

from ..data.dataset import CurriculumSampler


class VFRefineTrainer(Trainer):
    """Trainer for all three stages.

    Extra capabilities over the stock HF Trainer:
    * logs the four objective components (scaffold / command / coordinate /
      hierarchical-consistency) individually;
    * optionally replaces the random train sampler with the
      :class:`CurriculumSampler` (Stage III).
    """

    def __init__(self, *args, curriculum: Optional[Dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_cfg = curriculum or {}

    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.state.is_local_process_zero and self.control.should_log:
            logs: Dict[str, float] = {}
            for name in ("loss_scaffold", "loss_cmd", "loss_coord", "loss_hc"):
                val = getattr(outputs, name, None)
                if val is not None:
                    logs[name.replace("loss_", "")] = (
                        val.detach().float().mean().item()
                        if torch.is_tensor(val)
                        else float(val)
                    )
            if logs:
                self.log(logs)

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    def _get_train_sampler(self, *args, **kwargs):
        if self.curriculum_cfg.get("enabled", False) and hasattr(
            self.train_dataset, "complexity"
        ):
            return CurriculumSampler(
                self.train_dataset,
                total_epochs=int(self.args.num_train_epochs),
                min_fraction=float(self.curriculum_cfg.get("min_fraction", 0.3)),
                seed=int(self.args.seed),
            )
        return super()._get_train_sampler(*args, **kwargs)
