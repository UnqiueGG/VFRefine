# -*- coding: utf-8 -*-
# Vendored Qwen2.5-VL source (HuggingFace transformers v4.49.0).
#
# These files are the unmodified upstream reference implementations that the
# VFRefine vision encoder is adapted from. They are vendored so that the exact
# reference version is reproducible regardless of the installed transformers
# release.
from .configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from .modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
)

__all__ = [
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLVisionConfig",
    "Qwen2_5_VisionTransformerPretrainedModel",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel",
    "Qwen2_5_VLPreTrainedModel",
]
