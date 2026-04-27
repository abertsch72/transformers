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
OLMo pool configs — thin wrappers around Olmo2Config that cover the four
(block_type × qk_norm) architectural combinations produced by olmo-core training:

  reordered_norm + qk_norm=True  →  Olmo2Config (existing)
  reordered_norm + qk_norm=False →  Llama3ReorderedConfig  (NEW)
  default (pre-norm) + qk_norm=True  →  Olmo3PreorderConfig  (NEW)
  default (pre-norm) + qk_norm=False →  LlamaConfig (existing)
"""

from ..olmo2.configuration_olmo2 import Olmo2Config


class Olmo3PreorderConfig(Olmo2Config):
    r"""
    Config for models using a standard pre-norm (Llama-style) block with OLMo2-style
    QK norms on the attention projections.

    Weight key layout:
      model.layers.N.input_layernorm.weight          (pre-attention norm)
      model.layers.N.post_attention_layernorm.weight  (pre-FFN norm)
      model.layers.N.self_attn.q_norm.weight
      model.layers.N.self_attn.k_norm.weight

    Corresponds to olmo-core architectures: ``olmo2_*_preorder``,
    ``llama3_*_qknorm`` (OLMo lineage naming).
    """

    model_type = "olmo3_preorder"


class Llama3QKNormConfig(Olmo3PreorderConfig):
    r"""
    Config for models using a standard pre-norm block with QK norms — identical
    weight layout to :class:`Olmo3PreorderConfig`, but under the Llama lineage name.

    Corresponds to olmo-core architectures: ``llama3_*_qknorm``.
    """

    model_type = "llama3_qknorm"


class Llama3ReorderedConfig(Olmo2Config):
    r"""
    Config for models using a reordered-norm (OLMo2-style post-norm) block
    *without* QK norms on the attention projections.

    Weight key layout:
      model.layers.N.post_attention_layernorm.weight   (applied to attention output)
      model.layers.N.post_feedforward_layernorm.weight (applied to FFN output)
      (no q_norm / k_norm)

    Corresponds to olmo-core architectures: ``olmo2_*_noqk``,
    ``llama3_*_reordered``.
    """

    model_type = "llama3_reordered"


class Olmo2HeadwiseQKNormConfig(Olmo2Config):
    r"""
    Config for models using a reordered-norm (OLMo2-style post-norm) block with
    *headwise* QK norms — norms applied per attention head (size ``head_dim``)
    rather than across the full projection (size ``n_heads * head_dim``).

    Weight key layout:
      model.layers.N.post_attention_layernorm.weight
      model.layers.N.post_feedforward_layernorm.weight
      model.layers.N.self_attn.q_norm.weight  (shape: head_dim)
      model.layers.N.self_attn.k_norm.weight  (shape: head_dim)

    Corresponds to olmo-core architectures with ``use_head_qk_norm=True`` and a
    reordered-norm block (e.g. ``olmo2_7B`` with ``use_head_qk_norm=True``).
    """

    model_type = "olmo2_headwise_qknorm"


class Olmo3PreorderHeadwiseQKNormConfig(Olmo3PreorderConfig):
    r"""
    Config for models using a pre-norm (Llama-style) block with *headwise* QK
    norms — norms applied per attention head (size ``head_dim``) rather than
    across the full projection.

    Weight key layout:
      model.layers.N.input_layernorm.weight
      model.layers.N.post_attention_layernorm.weight
      model.layers.N.self_attn.q_norm.weight  (shape: head_dim)
      model.layers.N.self_attn.k_norm.weight  (shape: head_dim)

    Corresponds to olmo-core architectures with ``use_head_qk_norm=True`` and a
    default (pre-norm) block (e.g. ``qwen3_8B`` style).
    """

    model_type = "olmo3_preorder_headwise_qknorm"


__all__ = [
    "Llama3QKNormConfig",
    "Llama3ReorderedConfig",
    "Olmo2HeadwiseQKNormConfig",
    "Olmo3PreorderConfig",
    "Olmo3PreorderHeadwiseQKNormConfig",
]
