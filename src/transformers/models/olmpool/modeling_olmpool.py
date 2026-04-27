# Copyright 2025 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
OLMo pool model classes — three architectural variants that extend OLMo2.

  Olmo3Preorder*  — pre-norm block (input_layernorm / post_attention_layernorm) + QK norms
  Llama3QKNorm*   — identical architecture to Olmo3Preorder, Llama lineage name
  Llama3Reordered* — reordered-norm block (post_attention_layernorm /
                      post_feedforward_layernorm) without QK norms
"""

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.generic import TransformersKwargs
from ..llama.modeling_llama import LlamaAttention
from ..olmo2.modeling_olmo2 import (
    Olmo2Attention,
    Olmo2DecoderLayer,
    Olmo2ForCausalLM,
    Olmo2MLP,
    Olmo2Model,
    Olmo2PreTrainedModel,
    Olmo2RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_olmpool import (
    Llama3QKNormConfig,
    Llama3ReorderedConfig,
    Olmo2HeadwiseQKNormConfig,
    Olmo3PreorderConfig,
    Olmo3PreorderHeadwiseQKNormConfig,
)


# ---------------------------------------------------------------------------
# Olmo3Preorder — pre-norm (Llama-style) block with QK norms (OLMo lineage)
# ---------------------------------------------------------------------------


class Olmo3PreorderAttention(Olmo2Attention):
    """Olmo2Attention re-used verbatim; distinct class name for model registry."""

    pass


class Olmo3PreorderDecoderLayer(GradientCheckpointingLayer):
    """
    Pre-norm transformer block with QK norms.

    Norm placement (Llama-style):
      input_layernorm applied to x BEFORE attention.
      post_attention_layernorm applied to x BEFORE FFN.
    """

    def __init__(self, config: Olmo3PreorderConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Olmo3PreorderAttention(config=config, layer_idx=layer_idx)
        self.mlp = Olmo2MLP(config)
        self.input_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Olmo3PreorderModel(Olmo2Model):
    def __init__(self, config: Olmo3PreorderConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Olmo3PreorderDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )


@auto_docstring
class Olmo3PreorderForCausalLM(Olmo2ForCausalLM):
    config_class = Olmo3PreorderConfig
    _no_split_modules = ["Olmo3PreorderDecoderLayer"]

    def __init__(self, config: Olmo3PreorderConfig):
        super().__init__(config)
        self.model = Olmo3PreorderModel(config)
        self.post_init()


# ---------------------------------------------------------------------------
# Llama3QKNorm — identical architecture, Llama lineage name
# ---------------------------------------------------------------------------


class Llama3QKNormAttention(Olmo3PreorderAttention):
    """Olmo3PreorderAttention under the Llama3QKNorm class name."""

    pass


class Llama3QKNormDecoderLayer(Olmo3PreorderDecoderLayer):
    """Olmo3PreorderDecoderLayer under the Llama3QKNorm class name."""

    def __init__(self, config: Llama3QKNormConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Llama3QKNormAttention(config=config, layer_idx=layer_idx)


@auto_docstring
class Llama3QKNormModel(Olmo2Model):
    def __init__(self, config: Llama3QKNormConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Llama3QKNormDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )


@auto_docstring
class Llama3QKNormForCausalLM(Olmo2ForCausalLM):
    config_class = Llama3QKNormConfig
    _no_split_modules = ["Llama3QKNormDecoderLayer"]

    def __init__(self, config: Llama3QKNormConfig):
        super().__init__(config)
        self.model = Llama3QKNormModel(config)
        self.post_init()


# ---------------------------------------------------------------------------
# Llama3Reordered — reordered-norm block WITHOUT QK norms
# ---------------------------------------------------------------------------


class Llama3ReorderedAttention(LlamaAttention):
    """Standard LlamaAttention (no q_norm/k_norm) under the Llama3Reordered name."""

    pass


class Llama3ReorderedDecoderLayer(GradientCheckpointingLayer):
    """
    Reordered-norm transformer block without QK norms.

    Norm placement (OLMo2-style):
      post_attention_layernorm applied to attention OUTPUT before residual add.
      post_feedforward_layernorm applied to FFN OUTPUT before residual add.
    No q_norm or k_norm on the attention projections.
    """

    def __init__(self, config: Llama3ReorderedConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Llama3ReorderedAttention(config=config, layer_idx=layer_idx)
        self.mlp = Olmo2MLP(config)
        self.post_attention_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Llama3ReorderedModel(Olmo2Model):
    def __init__(self, config: Llama3ReorderedConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Llama3ReorderedDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )


@auto_docstring
class Llama3ReorderedForCausalLM(Olmo2ForCausalLM):
    config_class = Llama3ReorderedConfig
    _no_split_modules = ["Llama3ReorderedDecoderLayer"]

    def __init__(self, config: Llama3ReorderedConfig):
        super().__init__(config)
        self.model = Llama3ReorderedModel(config)
        self.post_init()


# ---------------------------------------------------------------------------
# Headwise QK norm attention — norm applied per-head after head reshape
# ---------------------------------------------------------------------------


class HeadwiseQKNormAttention(Olmo2Attention):
    """
    Attention where Q/K norms operate on each head independently (shape: head_dim)
    rather than across the full projection (shape: n_heads * head_dim).

    The norm is applied after projecting and reshaping to
    ``(batch, seq, n_heads, head_dim)`` — matching olmo-core's
    ``use_head_qk_norm=True`` behaviour.
    """

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        # Override: per-head norms sized to head_dim, not n_heads * head_dim.
        self.q_norm = Olmo2RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Olmo2RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project then reshape; apply headwise norm before transpose.
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Olmo2HeadwiseQKNorm — reordered-norm block with headwise QK norms
# ---------------------------------------------------------------------------


class Olmo2HeadwiseQKNormDecoderLayer(Olmo2DecoderLayer):
    """Reordered-norm decoder layer using headwise Q/K norms."""

    def __init__(self, config: Olmo2HeadwiseQKNormConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = HeadwiseQKNormAttention(config=config, layer_idx=layer_idx)


@auto_docstring
class Olmo2HeadwiseQKNormModel(Olmo2Model):
    def __init__(self, config: Olmo2HeadwiseQKNormConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Olmo2HeadwiseQKNormDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )


@auto_docstring
class Olmo2HeadwiseQKNormForCausalLM(Olmo2ForCausalLM):
    config_class = Olmo2HeadwiseQKNormConfig
    _no_split_modules = ["Olmo2HeadwiseQKNormDecoderLayer"]

    def __init__(self, config: Olmo2HeadwiseQKNormConfig):
        super().__init__(config)
        self.model = Olmo2HeadwiseQKNormModel(config)
        self.post_init()


# ---------------------------------------------------------------------------
# Olmo3PreorderHeadwiseQKNorm — pre-norm block with headwise QK norms
# ---------------------------------------------------------------------------


class Olmo3PreorderHeadwiseQKNormDecoderLayer(Olmo3PreorderDecoderLayer):
    """Pre-norm decoder layer using headwise Q/K norms."""

    def __init__(self, config: Olmo3PreorderHeadwiseQKNormConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = HeadwiseQKNormAttention(config=config, layer_idx=layer_idx)


@auto_docstring
class Olmo3PreorderHeadwiseQKNormModel(Olmo2Model):
    def __init__(self, config: Olmo3PreorderHeadwiseQKNormConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Olmo3PreorderHeadwiseQKNormDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )


@auto_docstring
class Olmo3PreorderHeadwiseQKNormForCausalLM(Olmo2ForCausalLM):
    config_class = Olmo3PreorderHeadwiseQKNormConfig
    _no_split_modules = ["Olmo3PreorderHeadwiseQKNormDecoderLayer"]

    def __init__(self, config: Olmo3PreorderHeadwiseQKNormConfig):
        super().__init__(config)
        self.model = Olmo3PreorderHeadwiseQKNormModel(config)
        self.post_init()


__all__ = [
    "HeadwiseQKNormAttention",
    "Llama3QKNormAttention",
    "Llama3QKNormDecoderLayer",
    "Llama3QKNormForCausalLM",
    "Llama3QKNormModel",
    "Llama3ReorderedAttention",
    "Llama3ReorderedDecoderLayer",
    "Llama3ReorderedForCausalLM",
    "Llama3ReorderedModel",
    "Olmo2HeadwiseQKNormDecoderLayer",
    "Olmo2HeadwiseQKNormForCausalLM",
    "Olmo2HeadwiseQKNormModel",
    "Olmo3PreorderAttention",
    "Olmo3PreorderDecoderLayer",
    "Olmo3PreorderForCausalLM",
    "Olmo3PreorderHeadwiseQKNormDecoderLayer",
    "Olmo3PreorderHeadwiseQKNormForCausalLM",
    "Olmo3PreorderHeadwiseQKNormModel",
    "Olmo3PreorderModel",
]
