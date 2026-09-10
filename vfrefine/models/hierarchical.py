# -*- coding: utf-8 -*-
"""Hierarchical SVG decoding components (paper Sec. "Hierarchical SVG Decoding").

The target SVG sequence is parsed into drawing primitives
``P = {(c_i, Theta_i)}`` where ``c_i in {M, L, H, V, C, Q, Z}`` is the path
command and ``Theta_i = (theta_{i,1}, ..., theta_{i,m_i})`` its quantised
coordinate parameters. Generation factorises as

    p(S_tgt | X) = prod_i p(c_i | P_{<i}, X) * p(Theta_i | c_i, P_{<i}, X)

with

    p(c_i | .)      = Softmax(W_cmd g_i + b_cmd)
    p(theta_ij | .) = Softmax(W_coord [h_ij (+) e(c_i)] + b_coord)

plus the hierarchical consistency constraint

    L_hc = sum_i sum_j || phi(h_ij) - psi(g_i) ||_2^2 .

To avoid duplicating the 152K-way output classifier, ``W_coord`` is factorised
as a command-conditioned fusion layer followed by the shared language-model
output embedding (``lm_head``).
"""

import torch
import torch.nn as nn

from ..constants import NUM_COMMANDS
from .configuration_vfrefine import VFRefineConfig
from .vision_encoder import VFRefinePatchMerger


class MultiScaleVisualProjector(nn.ModuleList):
    """The k vision-language fusion modules g^(k) of the multi-layer visual
    injection mechanism.

    Each g^(k) performs 2x2 spatial downsampling and dimensional projection of
    the tapped vision features F^(k), aligning them with the hidden space of
    the language model:  V^(k) = g^(k)(F^(k)) in R^{N_v x D_llm}.
    """

    def __init__(self, config: VFRefineConfig):
        vision_cfg = config.vision_config
        projectors = []
        for _ in vision_cfg.multiscale_tap_indexes:
            proj_cfg = VFRefineVisionConfigLike(
                vision_cfg, out_hidden_size=config.text_config.hidden_size
            )
            projectors.append(VFRefinePatchMerger(proj_cfg))
        super().__init__(projectors)


def VFRefineVisionConfigLike(vision_cfg, **overrides):
    """Shallow-copy a VFRefineVisionConfig overriding selected fields."""
    import copy

    cfg = copy.copy(vision_cfg)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class HierarchicalSVGHead(nn.Module):
    """Command head, command embedding and coordinate fusion layer.

    * ``cmd_head``:  W_cmd, predicts c_i over the 7 SVG path commands from the
      primitive-start contextual state g_i.
    * ``cmd_embed``: e(c), learned command embedding consumed by the
      command-conditioned coordinate decoder.
    * ``coord_fuse``: projects [h_ij (+) e(c_i)] back to the LLM hidden size;
      the shared ``lm_head`` then produces the vocabulary softmax over
      quantised coordinate tokens.
    * ``hc_phi`` / ``hc_psi``: linear projections of the hierarchical
      consistency constraint L_hc.
    """

    def __init__(self, config: VFRefineConfig):
        super().__init__()
        hidden = config.text_config.hidden_size
        self.cmd_head = nn.Linear(hidden, NUM_COMMANDS)
        self.cmd_embed = nn.Embedding(NUM_COMMANDS, hidden)
        self.coord_fuse = nn.Linear(2 * hidden, hidden)
        self.hc_phi = nn.Linear(hidden, hidden)
        self.hc_psi = nn.Linear(hidden, hidden)
        self.cmd_embed_scale = config.cmd_embed_scale

    def command_logits(self, g_i: torch.Tensor) -> torch.Tensor:
        """p(c_i | P_{<i}, X) = Softmax(W_cmd g_i + b_cmd)."""
        return self.cmd_head(g_i)

    def coordinate_hidden(self, h_ij: torch.Tensor, cmd_idx: torch.Tensor) -> torch.Tensor:
        """Fuse the parameter contextual state with the command embedding."""
        e = self.cmd_embed(cmd_idx) * self.cmd_embed_scale
        return self.coord_fuse(torch.cat([h_ij, e], dim=-1))

    def hc_loss(
        self,
        h_ij: torch.Tensor,
        g_i: torch.Tensor,
    ) -> torch.Tensor:
        """|| phi(h_ij) - psi(g_i) ||_2^2 (mean over all scalar parameters)."""
        diff = self.hc_phi(h_ij) - self.hc_psi(g_i)
        return diff.pow(2).sum(dim=-1).mean()
