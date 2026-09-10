# -*- coding: utf-8 -*-
"""VFRefine model.

Architecture (paper Fig. "Overview of VFRefine"):

    raster glyph(s) ---> VFRefine vision encoder --+--> 2x2 aggregated visual
                                                   |   tokens (input concat)
                                                   |
                                                   +--> multi-scale taps
                                                   |    F^{(1..3)} -> g^{(k)}
                                                   |    (residual layer
                                                   |     injection into
                                                   |     decoder layers
                                                   |     L_inj = {l1,l2,l3})
                                                   v
    SVG code <----- Qwen2.5-Coder-7B decoder <--- hierarchical SVG decoding
                    (token-level autoregression     heads (command / coord /
                     organised in command--param     consistency anchor)
                     groups)

The language backbone is the unmodified ``Qwen2Model``/``lm_head`` stack from
the HuggingFace Qwen2 implementation (Qwen2.5-Coder-7B shares the Qwen2
architecture), while the vision tower reuses the Qwen2.5-VL modules through
:class:`VFRefineVisionTransformerPretrainedModel`.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers.utils import logging

from ..constants import (
    IDX_TO_COMMAND,
    QWEN_IMAGE_PAD_ID,
    SVG_COMMAND_ARITY,
)
from .configuration_vfrefine import VFRefineConfig
from .hierarchical import HierarchicalSVGHead, MultiScaleVisualProjector
from .vision_encoder import VFRefineVisionTransformerPretrainedModel

logger = logging.get_logger(__name__)


@dataclass
class VFRefineCausalLMOutput(ModelOutput):
    """Output of :class:`VFRefineForConditionalGeneration`."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    loss_scaffold: Optional[torch.FloatTensor] = None
    loss_cmd: Optional[torch.FloatTensor] = None
    loss_coord: Optional[torch.FloatTensor] = None
    loss_hc: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class VFRefinePreTrainedModel(PreTrainedModel):
    config_class = VFRefineConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer", "Qwen2_5_VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class VFRefineModel(VFRefinePreTrainedModel):
    """The bare VFRefine model: vision tower + Qwen2 decoder stack."""

    def __init__(self, config: VFRefineConfig):
        super().__init__(config)
        self.visual = VFRefineVisionTransformerPretrainedModel(config.vision_config)
        self.multi_scale_projectors = MultiScaleVisualProjector(config)
        self.model = Qwen2Model(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value


class VFRefineForConditionalGeneration(VFRefinePreTrainedModel, GenerationMixin):
    """VFRefine with the LM head and the hierarchical SVG decoding heads."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: VFRefineConfig):
        super().__init__(config)
        self.visual = VFRefineVisionTransformerPretrainedModel(config.vision_config)
        self.multi_scale_projectors = MultiScaleVisualProjector(config)
        self.model = Qwen2Model(config.text_config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.hier_head = HierarchicalSVGHead(config)

        # residual layer-injection context: {decoder_layer_idx: (mask, values)}
        self._injection_ctx: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
        self._injection_hooks = []
        for layer_idx, tap_idx in zip(
            config.injection_layer_indexes, config.vision_config.multiscale_tap_indexes
        ):
            if layer_idx >= config.text_config.num_hidden_layers:
                raise ValueError(
                    f"injection layer {layer_idx} out of range for "
                    f"{config.text_config.num_hidden_layers} decoder layers"
                )
            hook = self.model.layers[layer_idx].register_forward_pre_hook(
                self._make_injection_hook(layer_idx), with_kwargs=True
            )
            self._injection_hooks.append(hook)

        self.post_init()

    # ------------------------------------------------------------------
    # embeddings / heads plumbing
    # ------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    # ------------------------------------------------------------------
    # multi-scale residual layer injection
    # ------------------------------------------------------------------
    def _make_injection_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            ctx = self._injection_ctx
            if ctx is None or layer_idx not in ctx:
                return None
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return None
            mask, values = ctx[layer_idx]
            # skip incremental decoding steps (sequence length 1) and any
            # call whose batch/sequence shape does not match the context.
            if hidden_states.shape[:2] != mask.shape:
                return None
            delta = torch.zeros_like(hidden_states)
            delta[mask] = values.to(delta.dtype)
            hidden_states = hidden_states + delta
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = hidden_states
            else:
                args = (hidden_states,) + args[1:]
            return args, kwargs

        return hook

    # ------------------------------------------------------------------
    # vision encoding
    # ------------------------------------------------------------------
    def encode_images(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the vision tower and the multi-scale projectors.

        Returns the aggregated visual tokens (scattered into the input
        embedding stream) and the projected multi-scale features V^{(k)} used
        for residual layer injection.
        """
        merged, taps = self.visual(pixel_values, image_grid_thw, return_multiscale=True)
        projected = [proj(tap) for proj, tap in zip(self.multi_scale_projectors, taps)]
        return merged, projected

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_token_id: int = QWEN_IMAGE_PAD_ID,
        # hierarchical decoding annotations (aligned with ``input_ids``;
        # -1 wherever the position is not of the corresponding kind)
        hier_cmd: Optional[torch.LongTensor] = None,
        hier_coord_cmd: Optional[torch.LongTensor] = None,
        hier_coord_prim: Optional[torch.LongTensor] = None,
        hier_prim_start: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, VFRefineCausalLMOutput]:
        return_dict = return_dict if return_dict is not None else True

        # reset the previous injection context (kept alive until here so that
        # gradient-checkpointed recomputation during backward still sees it)
        self._injection_ctx = None

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embeds, projected_taps = self.encode_images(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # register per-layer injection context
            bool_mask = input_ids == image_token_id
            self._injection_ctx = {}
            for layer_idx, values in zip(
                self.config.injection_layer_indexes, projected_taps
            ):
                self._injection_ctx[layer_idx] = (bool_mask, values)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        loss_scaffold = loss_cmd = loss_coord = loss_hc = None
        if labels is not None:
            (
                loss,
                loss_scaffold,
                loss_cmd,
                loss_coord,
                loss_hc,
            ) = self.compute_hierarchical_loss(
                hidden_states=hidden_states,
                logits=logits,
                labels=labels,
                hier_cmd=hier_cmd,
                hier_coord_cmd=hier_coord_cmd,
                hier_coord_prim=hier_coord_prim,
                hier_prim_start=hier_prim_start,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VFRefineCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            loss_scaffold=loss_scaffold,
            loss_cmd=loss_cmd,
            loss_coord=loss_coord,
            loss_hc=loss_hc,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    # ------------------------------------------------------------------
    # hierarchical decoding objective
    # ------------------------------------------------------------------
    def compute_hierarchical_loss(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        hier_cmd: Optional[torch.LongTensor],
        hier_coord_cmd: Optional[torch.LongTensor],
        hier_coord_prim: Optional[torch.LongTensor],
        hier_prim_start: Optional[torch.LongTensor],
    ):
        """L = l_scf * L_scaffold + l_cmd * L_cmd + l_coord * L_coord + l_hc * L_hc.

        All hierarchical annotation tensors are aligned with ``labels`` (i.e.
        they annotate the *target* token at each position). The standard
        next-token shift is applied uniformly.
        """
        cfg = self.config
        batch, seq = labels.shape
        device = hidden_states.device

        # states that *predict* the token at each label position
        hs = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_logits = logits[:, :-1, :]

        def shift(t):
            if t is None:
                return None
            return t[:, 1:]

        cmd_cls = shift(hier_cmd)
        coord_cmd = shift(hier_coord_cmd)
        coord_prim = shift(hier_coord_prim)
        prim_start = shift(hier_prim_start)

        ignore_index = -100

        # ---- scaffold token CE (everything that is neither a command nor a
        # coordinate parameter keeps the ordinary vocabulary softmax) ----
        scaffold_labels = shift_labels.clone()
        if cmd_cls is not None:
            scaffold_labels = scaffold_labels.masked_fill(cmd_cls >= 0, ignore_index)
        if coord_cmd is not None:
            scaffold_labels = scaffold_labels.masked_fill(coord_cmd >= 0, ignore_index)
        loss_scaffold = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            scaffold_labels.reshape(-1),
            ignore_index=ignore_index,
        )

        loss_cmd = hidden_states.new_zeros(())
        loss_coord = hidden_states.new_zeros(())
        loss_hc = hidden_states.new_zeros(())

        # ---- command prediction: p(c_i) = Softmax(W_cmd g_i + b_cmd) ----
        if cmd_cls is not None:
            cmd_mask = cmd_cls >= 0
            if cmd_mask.any():
                g_i = hs[cmd_mask]
                cmd_logits = self.hier_head.command_logits(g_i)
                loss_cmd = F.cross_entropy(cmd_logits, cmd_cls[cmd_mask])

        # ---- command-conditioned coordinate prediction ----
        # The Qwen2.5-Coder BPE splits numbers digit-by-digit, so one scalar
        # parameter spans several tokens: every token of the parameter carries
        # ``hier_coord_cmd`` (coord CE), while only its first token carries
        # ``hier_coord_prim`` (the per-parameter state h_ij used by L_hc).
        coord_mask = None
        if coord_cmd is not None:
            coord_mask = coord_cmd >= 0
            if coord_mask.any():
                h_ij = hs[coord_mask]
                fused = self.hier_head.coordinate_hidden(h_ij, coord_cmd[coord_mask])
                coord_logits = self.lm_head(fused)
                loss_coord = F.cross_entropy(coord_logits, shift_labels[coord_mask])

        # ---- hierarchical consistency: || phi(h_ij) - psi(g_i) ||^2 ----
        if (
            coord_prim is not None
            and (coord_prim >= 0).any()
            and prim_start is not None
            and cfg.lambda_hc > 0
        ):
            start_mask = prim_start >= 0
            param_mask = coord_prim >= 0
            if start_mask.any():
                # anchors are indexed by primitive id; the largest id may occur
                # at a parameter-less command (e.g. Z) that has no coordinate
                # tokens, so size the table from both annotation tensors.
                n_prim = int(
                    torch.max(
                        prim_start[start_mask].max(), coord_prim[param_mask].max()
                    ).item()
                ) + 1
                anchors = hs.new_zeros(batch, n_prim, hs.size(-1))
                b_idx, s_idx = torch.nonzero(start_mask, as_tuple=True)
                anchors[b_idx, prim_start[b_idx, s_idx]] = hs[b_idx, s_idx]
                bp_idx, sp_idx = torch.nonzero(param_mask, as_tuple=True)
                h_param = hs[bp_idx, sp_idx]
                g_i = anchors[bp_idx, coord_prim[bp_idx, sp_idx]]
                loss_hc = self.hier_head.hc_loss(h_param, g_i)

        loss = (
            cfg.lambda_scaffold * loss_scaffold
            + cfg.lambda_cmd * loss_cmd
            + cfg.lambda_coord * loss_coord
            + cfg.lambda_hc * loss_hc
        )
        return loss, loss_scaffold, loss_cmd, loss_coord, loss_hc

    # ------------------------------------------------------------------
    # structured generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def structured_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_token_id: int = QWEN_IMAGE_PAD_ID,
        eos_token_ids: Tuple[int, ...] = (151645, 151643),
        max_new_tokens: int = 4096,
        command_token_ids: Optional[Dict[int, int]] = None,
        first_command_token_ids: Optional[Dict[int, int]] = None,
        d_close_token_id: Optional[int] = None,
        digit_token_ids: Optional[Tuple[int, ...]] = None,
        max_digits: int = 3,
    ) -> torch.LongTensor:
        """Greedy command--parameter structured decoding.

        The prompt is expected to end with the SVG path prefix ``<path d="``.
        Decoding then alternates between

        * **command steps** — the closing quote of the path attribute is first
          checked through the ordinary LM head (ends the path); otherwise the
          command head selects c_i over the 7 drawing commands, and
        * **parameter steps** — exactly ``arity(c_i)`` scalar coordinates are
          emitted by the command-conditioned coordinate head. One scalar
          spans a leading space token plus up to ``max_digits`` digit tokens
          (the Qwen2.5-Coder BPE splits numbers digit-by-digit), so each
          scalar is decoded as a space token followed by a digit run.

        Args:
            command_token_ids: mapping command class -> token id of the
                ``" M"``-style (leading space) command token.
            first_command_token_ids: mapping command class -> token id of the
                command token directly after ``d="`` (no leading space).
            d_close_token_id: first token id of the ``"/></svg>"`` closing
                segment, which ends the structured phase.
            digit_token_ids: token ids of the ten decimal digits.

        Returns:
            The generated token ids (excluding the prompt), shape (1, L).
        """
        assert input_ids.size(0) == 1, "structured_generate is implemented for batch=1"
        assert command_token_ids is not None and first_command_token_ids is not None
        device = input_ids.device

        out = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_token_id=image_token_id,
            use_cache=True,
        )
        past = out.past_key_values
        last_hidden = out.last_hidden_state[:, -1:, :]

        generated: List[int] = []
        scaffold_mode = False
        expect_command = True
        is_first_command = True
        cur_cmd = -1
        params_left = 0
        budget = max_new_tokens

        def lm_argmax(hidden: torch.Tensor) -> int:
            return int(self.lm_head(hidden)[:, -1].argmax(-1).item())

        def coord_argmax(hidden: torch.Tensor, cmd: int) -> int:
            fused = self.hier_head.coordinate_hidden(
                hidden[:, -1], torch.tensor([cmd], device=device)
            )
            return int(self.lm_head(fused).argmax(-1).item())

        def step(next_id: int) -> None:
            nonlocal past, last_hidden, attention_mask
            generated.append(next_id)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(1, 1)], dim=1
            )
            out = self.forward(
                input_ids=torch.tensor([[next_id]], device=device),
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            last_hidden = out.last_hidden_state[:, -1:, :]

        while budget > 0:
            if scaffold_mode:
                # XML scaffolding is driven by the ordinary LM head
                next_id = lm_argmax(last_hidden)
                step(next_id)
                budget -= 1
                if next_id in eos_token_ids:
                    break
            elif expect_command:
                # does the LM head want to close the path attribute?
                if d_close_token_id is not None and lm_argmax(last_hidden) == d_close_token_id:
                    step(d_close_token_id)
                    budget -= 1
                    scaffold_mode = True
                    continue
                cur_cmd = int(
                    self.hier_head.command_logits(last_hidden[:, -1]).argmax(-1).item()
                )
                table = first_command_token_ids if is_first_command else command_token_ids
                step(table[cur_cmd])
                budget -= 1
                params_left = SVG_COMMAND_ARITY[IDX_TO_COMMAND[cur_cmd]]
                expect_command = params_left == 0
                is_first_command = False
            else:
                # one scalar parameter: leading (space) token + digit run
                step(coord_argmax(last_hidden, cur_cmd))
                budget -= 1
                n_digits = 0
                while n_digits < max_digits and budget > 0:
                    next_id = coord_argmax(last_hidden, cur_cmd)
                    if digit_token_ids is not None and next_id not in digit_token_ids:
                        break  # parameter finished early (untrained/defensive)
                    step(next_id)
                    budget -= 1
                    n_digits += 1
                params_left -= 1
                if params_left == 0:
                    expect_command = True

        return torch.tensor([generated], device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        image_grid_thw=None,
        **kwargs,
    ):
        # standard LLaVA-style behaviour: image tensors are only consumed by
        # the prefill step.
        if cache_position is not None and cache_position[0] > 0:
            pixel_values = None
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
            "cache_position": cache_position,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        return model_inputs
