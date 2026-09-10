# -*- coding: utf-8 -*-
"""Unified training entry point for the three-stage progressive scheme.

    # Stage I — SVG Syntax Pre-training (vision tower frozen)
    python scripts/train.py --config configs/stage1_ssp.yaml

    # Stage II — Raster-Vector Alignment
    python scripts/train.py --config configs/stage2_r2v.yaml

    # Stage III — Curriculum-based Refinement (few-shot/zero-shot switching)
    python scripts/train.py --config configs/stage3_refine.yaml

A later stage can initialise from an earlier checkpoint through
``model.init_from`` in the YAML (only matching keys are restored).
"""

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer, TrainingArguments, set_seed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vfrefine.data.collator import VFRefineCollator
from vfrefine.data.dataset import GlyphImageProcessor, R2VDataset, RefineDataset, SSPDataset
from vfrefine.training.trainer import VFRefineTrainer
from vfrefine.training.utils import build_model, load_config, save_model, set_trainable


def build_datasets(stage: int, data_cfg: dict):
    if stage == 1:
        train_ds = SSPDataset(data_cfg["train_jsonl"])
        eval_ds = SSPDataset(data_cfg["val_jsonl"]) if data_cfg.get("val_jsonl") else None
    elif stage == 2:
        train_ds = R2VDataset(data_cfg["train_split_dir"])
        eval_ds = R2VDataset(data_cfg["val_split_dir"]) if data_cfg.get("val_split_dir") else None
    else:
        train_ds = RefineDataset(
            data_cfg["train_split_dir"],
            num_references=data_cfg.get("num_references", 3),
            lambda_fewshot=data_cfg.get("lambda_fewshot", 0.5),
            p_augment=data_cfg.get("p_augment", 0.5),
            seed=data_cfg.get("seed", 0),
        )
        eval_ds = (
            RefineDataset(
                data_cfg["val_split_dir"],
                num_references=data_cfg.get("num_references", 3),
                lambda_fewshot=data_cfg.get("lambda_fewshot", 0.5),
                p_augment=0.0,
                seed=data_cfg.get("seed", 0) + 1,
            )
            if data_cfg.get("val_split_dir")
            else None
        )
    return train_ds, eval_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="resume from a trainer checkpoint dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stage = int(cfg.get("stage", 3))
    set_seed(int(cfg.get("seed", 42)))

    # ------------------------------ model ------------------------------
    model = build_model(
        cfg.get("model", {}),
        torch_dtype=torch.bfloat16 if cfg.get("training", {}).get("bf16", True) else torch.float32,
        attn_implementation=cfg.get("model", {}).get("attn_implementation", "sdpa"),
    )
    init_from = cfg.get("model", {}).get("init_from")
    if init_from:
        from safetensors.torch import load_file

        state = {}
        for shard in sorted(
            f for f in os.listdir(init_from) if f.endswith(".safetensors")
        ):
            state.update(load_file(os.path.join(init_from, shard)))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[init_from={init_from}] missing={len(missing)} unexpected={len(unexpected)}")
    set_trainable(model, stage)

    # ------------------------------ data -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.get("model", {}).get("tokenizer", cfg.get("model", {}).get("backbone"))
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_cfg = cfg.get("data", {})
    train_ds, eval_ds = build_datasets(stage, data_cfg)
    collator = VFRefineCollator(
        tokenizer,
        image_processor=GlyphImageProcessor(
            patch_size=model.config.vision_config.patch_size,
            temporal_patch_size=model.config.vision_config.temporal_patch_size,
            spatial_merge_size=model.config.vision_config.spatial_merge_size,
        ),
        max_length=data_cfg.get("max_length", 8192),
        spatial_merge_size=model.config.vision_config.spatial_merge_size,
    )
    print(f"train samples: {len(train_ds):,}" + (f" | eval samples: {len(eval_ds):,}" if eval_ds else ""))

    # ---------------------------- training -----------------------------
    tcfg = cfg.get("training", {})
    targs = TrainingArguments(
        output_dir=tcfg.get("output_dir", f"runs/stage{stage}"),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=tcfg.get("num_train_epochs", 2),
        max_steps=tcfg.get("max_steps", -1),
        learning_rate=tcfg.get("learning_rate", 1e-5),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        weight_decay=tcfg.get("weight_decay", 0.0),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        bf16=tcfg.get("bf16", True),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=tcfg.get("logging_steps", 10),
        save_steps=tcfg.get("save_steps", 2000),
        save_total_limit=tcfg.get("save_total_limit", 3),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=tcfg.get("eval_steps", 2000),
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 4),
        report_to=tcfg.get("report_to", ["tensorboard"]),
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = VFRefineTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        curriculum=tcfg.get("curriculum", {}),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    save_model(model, tokenizer, tcfg.get("output_dir", f"runs/stage{stage}"))


if __name__ == "__main__":
    main()
