# -*- coding: utf-8 -*-
"""VFRefine vision encoder.

Adapted from the Qwen2.5-VL vision transformer
(:class:`Qwen2_5_VisionTransformerPretrainedModel`, vendored under
:mod:`vfrefine.models.qwen2_5_vl`) with the following paper-specific changes:

1. **Native-resolution encoding** — glyph rasters are patchified on a
   non-overlapping P x P grid without resizing, preserving aspect ratio and
   high-frequency Bézier geometry.
2. **2D-RoPE** — decoupled row/column rotary position encoding, inherited
   unchanged from the Qwen2.5-VL implementation.
3. **Multi-scale window-based hybrid attention** — instead of a single global
   window size, each transformer block uses its own window schedule:
   blocks 1-6 cycle shifted windows over {2, 4, 8}, blocks 7-16 use window
   16, and blocks {7, 10, 13, 16} (1-indexed) use global attention.
4. **Multi-scale feature taps** — intermediate hidden states after blocks
   {6, 11, 16} are returned alongside the final features; they feed the
   residual layer-injection module of the language decoder.
5. **Spatial visual token aggregation** — adjacent 2x2 features are
   concatenated and projected to the LLM embedding space by a two-layer MLP,
   reducing the visual sequence length 4x.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_vfrefine import VFRefineVisionConfig
from .qwen2_5_vl.modeling_qwen2_5_vl import (
    QWEN2_5_VL_VISION_ATTENTION_CLASSES,
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLMLP,
    Qwen2_5_VLVisionBlock,
    Qwen2RMSNorm,
)

logger = logging.get_logger(__name__)


class VFRefinePatchMerger(nn.Module):
    """2x2 neighbour concatenation + two-layer MLP projection.

    v_{i,j} = [F_{2i,2j} (+) F_{2i+1,2j} (+) F_{2i,2j+1} (+) F_{2i+1,2j+1}]
    v'_{i,j} = W2 * act(W1 * v_{i,j} + b1) + b2
    """

    def __init__(self, config: VFRefineVisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, config.merger_hidden_size),
            nn.GELU(),
            nn.Linear(config.merger_hidden_size, config.out_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, hidden) where seq_len groups spatial_merge_unit
        # consecutive patches that belong to the same 2x2 neighbourhood.
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VFRefineVisionTransformerPretrainedModel(PreTrainedModel):
    """Hybrid-attention dynamic-resolution vision encoder of VFRefine."""

    config_class = VFRefineVisionConfig
    base_model_prefix = "visual"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]

    def __init__(self, config: VFRefineVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.multiscale_tap_indexes = config.multiscale_tap_indexes

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        attn_implementation = getattr(config, "_attn_implementation", None) or "sdpa"
        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VLVisionBlock(config, attn_implementation)
                for _ in range(config.depth)
            ]
        )
        self.merger = VFRefinePatchMerger(config)
        self.gradient_checkpointing = False

        # validate the attention schedule up-front
        for idx in range(config.depth):
            w = config.window_size_for_block(idx)
            if idx in self.fullatt_block_indexes:
                assert w is None, f"block {idx}: expected global attention"
        assert all(
            0 <= t < config.depth for t in self.multiscale_tap_indexes
        ), "multi-scale tap index out of range"

    # ------------------------------------------------------------------
    # positional encoding (identical to Qwen2.5-VL: decoupled 2D-RoPE whose
    # attention score depends only on relative displacement (i-i', j-j'))
    # ------------------------------------------------------------------
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    # ------------------------------------------------------------------
    # windowing: generalises Qwen2.5-VL's single-window partitioning to
    # per-layer window sizes and Swin-style shifted windows.
    # ------------------------------------------------------------------
    def get_window_index(
        self,
        grid_thw: torch.Tensor,
        window_size: int,
        shift: int = 0,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Partition the merged token grid into ``window_size`` x ``window_size``
        windows.

        Args:
            grid_thw: (num_images, 3) temporal/height/width of the *patch* grid.
            window_size: window side length in merged (2x2-aggregated) units.
            shift: cyclic shift (in merged units) applied before partitioning,
                implementing the shifted-window mechanism. With ``shift > 0``
                windows that wrap around the border mix distant tokens, which
                enlarges the effective receptive field (Swin-style shift
                without an attention mask).

        Returns:
            window_index: permutation that groups tokens of the same window
                together (in merge-unit resolution).
            cu_window_seqlens: cumulative window lengths in *patch* tokens.
        """
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = window_size  # already in merged-grid units

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            if shift > 0:
                # cyclic shift of the token grid before partitioning
                index = torch.roll(index, shifts=(shift, shift), dims=(-2, -1))
            pad_h = (
                vit_merger_window_size - llm_grid_h % vit_merger_window_size
            ) % vit_merger_window_size
            pad_w = (
                vit_merger_window_size - llm_grid_w % vit_merger_window_size
            ) % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def _attention_schedule(
        self, grid_thw: torch.Tensor, device: torch.device
    ) -> List[Tuple[Optional[torch.Tensor], torch.Tensor]]:
        """Precompute, for every block, the (window_index, cu_seqlens) pair.

        ``window_index is None`` denotes global attention, in which case
        ``cu_seqlens`` are the per-image cumulative lengths.
        """
        cu_seqlens_full = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens_full = F.pad(cu_seqlens_full, (1, 0), value=0)

        cache: Dict[Tuple[int, int], Tuple[torch.Tensor, List[int]]] = {}
        schedule: List[Tuple[Optional[torch.Tensor], torch.Tensor]] = []
        for block_idx in range(self.config.depth):
            w = self.config.window_size_for_block(block_idx)
            if w is None:
                schedule.append((None, cu_seqlens_full))
                continue
            shift = w // 2 if self.config.use_shifted_window(block_idx) else 0
            key = (w, shift)
            if key not in cache:
                window_index, cu_window_seqlens = self.get_window_index(
                    grid_thw, window_size=w, shift=shift
                )
                cu_window_seqlens_t = torch.tensor(
                    cu_window_seqlens, device=device, dtype=torch.int32
                )
                cu_window_seqlens_t = torch.unique_consecutive(cu_window_seqlens_t)
                cache[key] = (window_index.to(device), cu_window_seqlens_t)
            schedule.append(cache[key])
        return schedule

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        return_multiscale: bool = True,
    ):
        """
        Args:
            hidden_states: (total_patch_tokens, C * T * P * P) flattened patches.
            grid_thw: (num_images, 3) patch-grid dimensions per image.
            return_multiscale: also return intermediate features after the
                tapped blocks (used by residual layer injection).

        Returns:
            merged: (total_merged_tokens, out_hidden_size) aggregated visual
                tokens in canonical raster order.
            taps (optional): list of (total_patch_tokens, hidden_size)
                intermediate features, one per entry of
                ``config.multiscale_tap_indexes``, also in raster order.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        schedule = self._attention_schedule(grid_thw, hidden_states.device)

        taps: List[torch.Tensor] = []
        # tokens are kept in canonical raster order between blocks; each block
        # permutes into its own window order and back.
        for layer_num, blk in enumerate(self.blocks):
            window_index, cu_seqlens_now = schedule[layer_num]
            if window_index is not None:
                hidden_states = hidden_states.reshape(
                    seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
                )
                hidden_states = hidden_states[window_index, :, :]
                hidden_states = hidden_states.reshape(seq_len, -1)
                pos_emb_now = tuple(
                    pe.reshape(
                        seq_len // self.spatial_merge_unit,
                        self.spatial_merge_unit,
                        -1,
                    )[window_index, :, :].reshape(seq_len, -1)
                    for pe in position_embeddings
                )
            else:
                pos_emb_now = position_embeddings

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, pos_emb_now
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    position_embeddings=pos_emb_now,
                )

            # restore canonical raster order
            if window_index is not None:
                reverse_indices = torch.argsort(window_index)
                hidden_states = hidden_states.reshape(
                    seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
                )
                hidden_states = hidden_states[reverse_indices, :, :]
                hidden_states = hidden_states.reshape(seq_len, -1)

            if return_multiscale and layer_num in self.multiscale_tap_indexes:
                taps.append(hidden_states)

        merged = self.merger(hidden_states)
        if return_multiscale:
            return merged, taps
        return merged
