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

from typing import Optional

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.generic import TransformersKwargs
from ..llama.modeling_llama import LlamaAttention
from ..olmo2.modeling_olmo2 import (
    Olmo2Attention,
    Olmo2ForCausalLM,
    Olmo2MLP,
    Olmo2Model,
    Olmo2PreTrainedModel,
    Olmo2RMSNorm,
)
from .configuration_olmpool import Llama3QKNormConfig, Llama3ReorderedConfig, Olmo3PreorderConfig


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


__all__ = [
    "Llama3QKNormAttention",
    "Llama3QKNormDecoderLayer",
    "Llama3QKNormForCausalLM",
    "Llama3QKNormModel",
    "Llama3ReorderedAttention",
    "Llama3ReorderedDecoderLayer",
    "Llama3ReorderedForCausalLM",
    "Llama3ReorderedModel",
    "Olmo3PreorderAttention",
    "Olmo3PreorderDecoderLayer",
    "Olmo3PreorderForCausalLM",
    "Olmo3PreorderModel",
]
