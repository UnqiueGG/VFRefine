# -*- coding: utf-8 -*-
"""Configuration for VFRefine.

VFRefine couples a hybrid-attention dynamic-resolution vision encoder with a
Qwen2.5-Coder-7B language backbone. Default values follow the architecture
tables of the paper:

* Vision encoder: 16 transformer blocks, hidden 1280, intermediate 3456,
  16 heads, 2D-RoPE, multi-scale window-based hybrid attention
  (shallow shifted windows {2, 4, 8} applied cyclically, deep window 16,
  global attention at blocks {7, 10, 13, 16}), 2x2 spatial token aggregation.
* Language model: Qwen2ForCausalLM architecture, 28 layers, hidden 3584,
  28 attention heads, 4 KV heads, FFN 18928, RoPE theta 1e6, vocab 152064.
"""

from transformers import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class VFRefineVisionConfig(PretrainedConfig):
    """Configuration of the VFRefine hybrid-attention vision encoder."""

    model_type = "vfrefine_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=16,
        hidden_size=1280,
        hidden_act="silu",
        intermediate_size=3456,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        # hybrid attention schedule -------------------------------------
        # shallow blocks 1-6: shifted windows cycled over {2, 4, 8}
        shallow_window_sizes=(2, 4, 8),
        num_shallow_blocks=6,
        # deep blocks 7-16: window 16, global attention at {7, 10, 13, 16}
        deep_window_size=16,
        fullatt_block_indexes=(6, 9, 12, 15),  # 0-indexed blocks {7,10,13,16}
        # multi-scale feature taps used by layer injection (0-indexed
        # blocks {6, 11, 16} -> local geometry / mid structure / global)
        multiscale_tap_indexes=(5, 10, 15),
        out_hidden_size=3584,
        merger_hidden_size=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.shallow_window_sizes = tuple(shallow_window_sizes)
        self.num_shallow_blocks = num_shallow_blocks
        self.deep_window_size = deep_window_size
        self.fullatt_block_indexes = tuple(fullatt_block_indexes)
        self.multiscale_tap_indexes = tuple(multiscale_tap_indexes)
        self.out_hidden_size = out_hidden_size
        # hidden width of the 2-layer aggregation MLP; defaults to the
        # Qwen2.5-VL convention (4 * hidden after 2x2 concatenation)
        self.merger_hidden_size = (
            merger_hidden_size
            if merger_hidden_size is not None
            else hidden_size * (spatial_merge_size**2)
        )
        self.initializer_range = initializer_range

    def window_size_for_block(self, block_idx):
        """Return the window size (in merged-grid units) for a block.

        Returns ``None`` for blocks that use global (full) attention.
        """
        if block_idx in self.fullatt_block_indexes:
            return None
        if block_idx < self.num_shallow_blocks:
            cycle = self.shallow_window_sizes
            return cycle[block_idx % len(cycle)]
        return self.deep_window_size

    def use_shifted_window(self, block_idx):
        """Shifted-window flag: shallow blocks alternate plain/shifted."""
        return block_idx < self.num_shallow_blocks and (block_idx % 2 == 1)


class VFRefineConfig(PretrainedConfig):
    """Top-level VFRefine configuration.

    Attributes:
        vision_config (:class:`VFRefineVisionConfig`): visual encoder config.
        text_config (:class:`~transformers.Qwen2Config`): Qwen2.5-Coder-7B
            backbone configuration.
        injection_layer_indexes (tuple): decoder layers (0-indexed) into which
            the multi-scale visual features are residually injected
            (paper Sec. "Residual Layer Injection", set L_inj = {l1, l2, l3}).
        lambda_cmd / lambda_coord / lambda_hc: weights of the hierarchical
            decoding objective
            L = lambda_cmd * L_cmd + lambda_coord * L_coord + lambda_hc * L_hc.
        lambda_scaffold: weight of the standard token-level cross-entropy on
            the XML scaffolding tokens (everything that is neither a path
            command nor a coordinate parameter).
    """

    model_type = "vfrefine"
    sub_configs = {
        "vision_config": VFRefineVisionConfig,
        "text_config": Qwen2Config,
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        injection_layer_indexes=(4, 9, 14),
        lambda_cmd=1.0,
        lambda_coord=1.0,
        lambda_hc=0.1,
        lambda_scaffold=1.0,
        cmd_embed_scale=1.0,
        initializer_range=0.02,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = VFRefineVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = VFRefineVisionConfig(**vision_config)

        if text_config is None:
            # Qwen2.5-Coder-7B backbone (paper: Language Modeling config table)
            text_config = Qwen2Config(
                vocab_size=152064,
                hidden_size=3584,
                intermediate_size=18928,
                num_hidden_layers=28,
                num_attention_heads=28,
                num_key_value_heads=4,
                hidden_act="silu",
                max_position_embeddings=131072,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=True,
                tie_word_embeddings=False,
                rope_theta=1000000.0,
                use_sliding_window=False,
                sliding_window=131072,
                max_window_layers=28,
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151643,
            )
        elif isinstance(text_config, dict):
            text_config = Qwen2Config(**text_config)

        self.vision_config = vision_config
        self.text_config = text_config
        self.injection_layer_indexes = tuple(injection_layer_indexes)
        assert len(self.injection_layer_indexes) == len(
            self.vision_config.multiscale_tap_indexes
        ), "one decoder injection layer is required per multi-scale visual tap"
        self.lambda_cmd = lambda_cmd
        self.lambda_coord = lambda_coord
        self.lambda_hc = lambda_hc
        self.lambda_scaffold = lambda_scaffold
        self.cmd_embed_scale = cmd_embed_scale
        self.initializer_range = initializer_range

        # keep top-level scalar mirrors so that generic HF utilities that read
        # config.hidden_size / config.vocab_size keep working
        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size
        super().__init__(
            bos_token_id=self.text_config.bos_token_id,
            eos_token_id=self.text_config.eos_token_id,
            **kwargs,
        )

    @property
    def llm_hidden_size(self):
        return self.text_config.hidden_size
