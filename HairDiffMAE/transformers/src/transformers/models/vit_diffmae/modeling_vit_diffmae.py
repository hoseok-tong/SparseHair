#################### Enocde only visible patches
# coding=utf-8
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ViT MAE (masked autoencoder) model."""


import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_vit_diffmae import ViTDiffMAEConfig
from .embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ViTDiffMAEConfig"
_CHECKPOINT_FOR_DOC = "facebook/vit-mae-base"

VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/vit-mae-base",
    # See all ViTDiffMAE models at https://huggingface.co/models?filter=vit_mae
]


@dataclass
class ViTDiffMAEModelOutput(ModelOutput):
    """
    Class for ViTDiffMAEModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    noise: torch.FloatTensor = None


@dataclass
class ViTDiffMAEDecoderOutput(ModelOutput):
    """
    Class for ViTDiffMAEDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ViTDiffMAEForPreTrainingOutput(ModelOutput):
    """
    Class for ViTDiffMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    noise: torch.FloatTensor = None


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ViTDiffMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTDiffMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches**0.5), add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore, noise

    def forward(self, pixel_values, noise=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore, noise = self.random_masking(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore, noise


class ViTDiffMAEPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention ViT->ViTDiffMAE
class ViTDiffMAESelfAttention(nn.Module):
    def __init__(self, config: ViTDiffMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->ViTDiffMAE
class ViTDiffMAESelfOutput(nn.Module):
    """
    The residual connection is defined in ViTDiffMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTDiffMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->ViTDiffMAE
class ViTDiffMAEAttention(nn.Module):
    def __init__(self, config: ViTDiffMAEConfig) -> None:
        super().__init__()
        self.attention = ViTDiffMAESelfAttention(config)
        self.output = ViTDiffMAESelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate ViT->ViTDiffMAE
class ViTDiffMAEIntermediate(nn.Module):
    def __init__(self, config: ViTDiffMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput ViT->ViTDiffMAE
class ViTDiffMAEOutput(nn.Module):
    def __init__(self, config: ViTDiffMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->ViTDiffMAE
class ViTDiffMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTDiffMAEConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTDiffMAEAttention(config)
        self.intermediate = ViTDiffMAEIntermediate(config)
        self.output = ViTDiffMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViTDiffMAE, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViTDiffMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->ViTDiffMAE
class ViTDiffMAEEncoder(nn.Module):
    def __init__(self, config: ViTDiffMAEConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTDiffMAELayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTDiffMAEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTDiffMAEConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


VIT_MAE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTDiffMAEConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_MAE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare ViTDiffMAE Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_MAE_START_DOCSTRING,
)
class ViTDiffMAEModel(ViTDiffMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTDiffMAEEmbeddings(config)
        self.encoder = ViTDiffMAEEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ViTDiffMAEModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_noise: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTDiffMAEModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTDiffMAEModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTDiffMAEModel.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, mask, ids_restore, noise = self.embeddings(pixel_values, noise=noise)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)


        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return ViTDiffMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            noise=noise
        )


class ViTDiffMAEDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTDiffMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights(num_patches)

        #########################
        block_out_channels = 256    # 224
        time_embed_dim = 256 * 4    # 224*4
        flip_sin_to_cos = True
        freq_shift = 0

        self.time_proj = Timesteps(block_out_channels, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(block_out_channels, time_embed_dim)

        self.time_layers = nn.ModuleList(
            [nn.Linear(time_embed_dim, config.decoder_hidden_size, bias=True) for _ in range(config.decoder_num_hidden_layers)]
        )
        #########################


    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        pixel_noise, # img_noise
        timesteps,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):

        #########################
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)
        #########################

        # embed tokens
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        # print("x", x.shape)                         # torch.Size([64, 65, 512])
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # print("mask_tokens", mask_tokens.shape)     # torch.Size([64, 192, 512])
        # mask_tokens = self.decoder_embed(mask_tokens)   # 하든 안하든 loss 가 안내려감. 의미 없음 vitmae에서는 안함.
        # print("mask_tokens", mask_tokens.shape)     # torch.Size([64, 192, 512])
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # print("x_", x_.shape)                       # torch.Size([64, 256, 512])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # print("x_", x_.shape)                       # torch.Size([64, 256, 512])
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # print("x", x.shape)                         # torch.Size([64, 257, 512])


        # add pos embed
        hidden_states = x + self.decoder_pos_embed
        # print("self.decoder_pos_embed", self.decoder_pos_embed.shape)   # torch.Size([1, 257, 512])

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, (layer_module, time_layer_module) in enumerate(zip(self.decoder_layers, self.time_layers)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
                print("hi")
            else:
                #########################
                layer_time = time_layer_module(emb)
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)
                layer_outputs = (layer_outputs[0] + layer_time.unsqueeze(1), )
                #########################
            # print("layer_outputs[0].shape", layer_outputs[0].shape) # torch.Size([32, 257, 512])
            # print("layer_outputs[1].shape", layer_outputs[1].shape)
            # print("hidden_states.shape", hidden_states.shape)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return ViTDiffMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings(
    """The ViTDiffMAE Model transformer with the decoder on top for self-supervised pre-training.

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>

    """,
    VIT_MAE_START_DOCSTRING,
)
class ViTDiffMAEForPreTraining(ViTDiffMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = ViTDiffMAEModel(config)
        self.decoder = ViTDiffMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.vit.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # sanity checks
        if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_one_direction = pixel_values.shape[2] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
        # sanity check
        if num_patches_one_direction**2 != patchified_pixel_values.shape[1]:
            raise ValueError("Make sure that the number of patches can be squared")

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_one_direction,
            num_patches_one_direction,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_one_direction * patch_size,
            num_patches_one_direction * patch_size,
        )
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask, baldnessmap=None):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values)
        if self.config.norm_pix_loss:   # Set self.config.norm_pix_loss == False
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        if baldnessmap is not None:
            baldness_patches = self.patchify(baldnessmap)
            target = target * baldness_patches
            pred = pred * baldness_patches

        loss = (pred - target) ** 2 # pred: [batch_size, num_patches, patch_size**2 * num_channels].
        loss = loss.mean(dim=-1)  # [N, L], # [batch_size, num_patches] mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on masked patches
        return loss


    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ViTDiffMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_noise: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.FloatTensor] = None,
        baldnessmap: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        added_noise: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTDiffMAEForPreTrainingOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTDiffMAEForPreTraining
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTDiffMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> mask = outputs.mask
        >>> ids_restore = outputs.ids_restore
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,   # img_clean
            pixel_noise,    # img_noise
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask
        noise = outputs.noise


        #########################
        timesteps = timesteps * torch.ones(pixel_values.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        #########################

        decoder_outputs = self.decoder(latent, pixel_noise, timesteps, ids_restore)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        loss = self.forward_loss(pixel_values, logits, mask, baldnessmap)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ViTDiffMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            noise=outputs.noise,
        )
#################### Enocde only visible patches




# # coding=utf-8
# # Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """ PyTorch ViT MAE (masked autoencoder) model."""


# import collections.abc
# import math
# from copy import deepcopy
# from dataclasses import dataclass
# from typing import Optional, Set, Tuple, Union

# import numpy as np
# import torch
# import torch.utils.checkpoint
# from torch import nn

# from ...activations import ACT2FN
# from ...modeling_outputs import BaseModelOutput
# from ...modeling_utils import PreTrainedModel
# from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
# from ...utils import (
#     ModelOutput,
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     logging,
#     replace_return_docstrings,
# )
# from .configuration_vit_diffmae import ViTDiffMAEConfig
# from .embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps


# logger = logging.get_logger(__name__)

# _CONFIG_FOR_DOC = "ViTDiffMAEConfig"
# _CHECKPOINT_FOR_DOC = "facebook/vit-mae-base"

# VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST = [
#     "facebook/vit-mae-base",
#     # See all ViTDiffMAE models at https://huggingface.co/models?filter=vit_mae
# ]


# @dataclass
# class ViTDiffMAEModelOutput(ModelOutput):
#     """
#     Class for ViTDiffMAEModel's outputs, with potential hidden states and attentions.

#     Args:
#         last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
#             Sequence of hidden-states at the output of the last layer of the model.
#         mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
#             Tensor indicating which patches are masked (1) and which are not (0).
#         ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             Tensor containing the original index of the (shuffled) masked patches.
#         hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
#             shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
#             plus the initial embedding outputs.
#         attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
#             sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
#             the self-attention heads.
#     """

#     last_hidden_state: torch.FloatTensor = None
#     last_hidden_state_noise: torch.FloatTensor = None
#     mask: torch.LongTensor = None
#     ids_restore: torch.LongTensor = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None
#     noise: torch.FloatTensor = None


# @dataclass
# class ViTDiffMAEDecoderOutput(ModelOutput):
#     """
#     Class for ViTDiffMAEDecoder's outputs, with potential hidden states and attentions.

#     Args:
#         logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
#             Pixel reconstruction logits.
#         hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
#             shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
#             plus the initial embedding outputs.
#         attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
#             sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
#             the self-attention heads.
#     """

#     logits: torch.FloatTensor = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None


# @dataclass
# class ViTDiffMAEForPreTrainingOutput(ModelOutput):
#     """
#     Class for ViTDiffMAEForPreTraining's outputs, with potential hidden states and attentions.

#     Args:
#         loss (`torch.FloatTensor` of shape `(1,)`):
#             Pixel reconstruction loss.
#         logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
#             Pixel reconstruction logits.
#         mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
#             Tensor indicating which patches are masked (1) and which are not (0).
#         ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             Tensor containing the original index of the (shuffled) masked patches.
#         hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
#             shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
#             plus the initial embedding outputs.
#         attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
#             sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
#             the self-attention heads.
#     """

#     loss: Optional[torch.FloatTensor] = None
#     logits: torch.FloatTensor = None
#     mask: torch.LongTensor = None
#     ids_restore: torch.LongTensor = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None
#     noise: torch.FloatTensor = None


# def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
#     """
#     Create 2D sin/cos positional embeddings.

#     Args:
#         embed_dim (`int`):
#             Embedding dimension.
#         grid_size (`int`):
#             The grid height and width.
#         add_cls_token (`bool`, *optional*, defaults to `False`):
#             Whether or not to add a classification (CLS) token.

#     Returns:
#         (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
#         position embeddings (with or without classification token)
#     """
#     grid_h = np.arange(grid_size, dtype=np.float32)
#     grid_w = np.arange(grid_size, dtype=np.float32)
#     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
#     grid = np.stack(grid, axis=0)

#     grid = grid.reshape([2, 1, grid_size, grid_size])
#     pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
#     if add_cls_token:
#         pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
#     return pos_embed


# def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
#     if embed_dim % 2 != 0:
#         raise ValueError("embed_dim must be even")

#     # use half of dimensions to encode grid_h
#     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
#     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

#     emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
#     return emb


# def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
#     """
#     embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
#     """
#     if embed_dim % 2 != 0:
#         raise ValueError("embed_dim must be even")

#     omega = np.arange(embed_dim // 2, dtype=float)
#     omega /= embed_dim / 2.0
#     omega = 1.0 / 10000**omega  # (D/2,)

#     pos = pos.reshape(-1)  # (M,)
#     out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

#     emb_sin = np.sin(out)  # (M, D/2)
#     emb_cos = np.cos(out)  # (M, D/2)

#     emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
#     return emb


# class ViTDiffMAEEmbeddings(nn.Module):
#     """
#     Construct the CLS token, position and patch embeddings.

#     """

#     def __init__(self, config):
#         super().__init__()

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
#         self.patch_embeddings = ViTDiffMAEPatchEmbeddings(config)
#         self.num_patches = self.patch_embeddings.num_patches
#         # fixed sin-cos embedding
#         self.position_embeddings = nn.Parameter(
#             torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
#         )
#         self.config = config
#         self.initialize_weights()

#     def initialize_weights(self):
#         # initialize (and freeze) position embeddings by sin-cos embedding
#         pos_embed = get_2d_sincos_pos_embed(
#             self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches**0.5), add_cls_token=True
#         )
#         self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
#         w = self.patch_embeddings.projection.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

#     def random_masking(self, sequence, sequence_noise, noise=None):
#         """
#         Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
#         noise.

#         Args:
#             sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
#             noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
#                 mainly used for testing purposes to control randomness and maintain the reproducibility
#         """
#         batch_size, seq_length, dim = sequence.shape
#         len_keep = int(seq_length * (1 - self.config.mask_ratio))

#         if noise is None:
#             noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#         ids_restore = torch.argsort(ids_shuffle, dim=1)

#         # keep the first subset (visible patches)
#         ids_keep = ids_shuffle[:, :len_keep]
#         sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

#         #########################
#         # remove (will be masked) (unvisible patches)
#         ids_noise = ids_shuffle[:, len_keep:]
#         sequence_unmasked_noise = torch.gather(sequence_noise, dim=1, index=ids_noise.unsqueeze(-1).repeat(1, 1, dim))
#         #########################

#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([batch_size, seq_length], device=sequence.device)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask
#         mask = torch.gather(mask, dim=1, index=ids_restore)

#         return sequence_unmasked, sequence_unmasked_noise, mask, ids_restore, noise

#     def forward(self, pixel_values, pixel_noise, noise=None):
#         batch_size, num_channels, height, width = pixel_values.shape
#         embeddings = self.patch_embeddings(pixel_values)

#         # add position embeddings w/o cls token
#         embeddings = embeddings + self.position_embeddings[:, 1:, :]

#         #########################
#         embeddings_noise = self.patch_embeddings(pixel_noise)
#         embeddings_noise = embeddings_noise + self.position_embeddings[:, 1:, :]
#         #########################

#         # masking: length -> length * config.mask_ratio
#         embeddings, embeddings_noise, mask, ids_restore, noise = self.random_masking(embeddings, embeddings_noise, noise)
#         # embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

#         # append cls token
#         cls_token = self.cls_token + self.position_embeddings[:, :1, :]
#         cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
#         embeddings = torch.cat((cls_tokens, embeddings), dim=1)

#         ####################
#         # embeddings_noise = torch.cat((cls_tokens, embeddings_noise), dim=1)
#         ####################
#         return embeddings, embeddings_noise, mask, ids_restore, noise


# class ViTDiffMAEPatchEmbeddings(nn.Module):
#     """
#     This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
#     `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
#     Transformer.
#     """

#     def __init__(self, config):
#         super().__init__()
#         image_size, patch_size = config.image_size, config.patch_size
#         num_channels, hidden_size = config.num_channels, config.hidden_size
#         image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
#         patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
#         num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.num_channels = num_channels
#         self.num_patches = num_patches

#         self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

#     def forward(self, pixel_values):
#         batch_size, num_channels, height, width = pixel_values.shape
#         if num_channels != self.num_channels:
#             raise ValueError(
#                 "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
#             )
#         if height != self.image_size[0] or width != self.image_size[1]:
#             raise ValueError(
#                 f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
#             )
#         x = self.projection(pixel_values).flatten(2).transpose(1, 2)
#         return x


# # Copied from transformers.models.vit.modeling_vit.ViTSelfAttention ViT->ViTDiffMAE
# class ViTDiffMAESelfAttention(nn.Module):
#     def __init__(self, config: ViTDiffMAEConfig) -> None:
#         super().__init__()
#         if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
#             raise ValueError(
#                 f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
#                 f"heads {config.num_attention_heads}."
#             )

#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

#     def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(
#         self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
#     ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#         mixed_query_layer = self.query(hidden_states)

#         key_layer = self.transpose_for_scores(self.key(hidden_states))
#         value_layer = self.transpose_for_scores(self.value(hidden_states))
#         query_layer = self.transpose_for_scores(mixed_query_layer)

#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.functional.softmax(attention_scores, dim=-1)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)

#         # Mask heads if we want to
#         if head_mask is not None:
#             attention_probs = attention_probs * head_mask

#         context_layer = torch.matmul(attention_probs, value_layer)

#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(new_context_layer_shape)

#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

#         return outputs


# # Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->ViTDiffMAE
# class ViTDiffMAESelfOutput(nn.Module):
#     """
#     The residual connection is defined in ViTDiffMAELayer instead of here (as is the case with other models), due to the
#     layernorm applied before each block.
#     """

#     def __init__(self, config: ViTDiffMAEConfig) -> None:
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)

#         return hidden_states


# # Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->ViTDiffMAE
# class ViTDiffMAEAttention(nn.Module):
#     def __init__(self, config: ViTDiffMAEConfig) -> None:
#         super().__init__()
#         self.attention = ViTDiffMAESelfAttention(config)
#         self.output = ViTDiffMAESelfOutput(config)
#         self.pruned_heads = set()

#     def prune_heads(self, heads: Set[int]) -> None:
#         if len(heads) == 0:
#             return
#         heads, index = find_pruneable_heads_and_indices(
#             heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
#         )

#         # Prune linear layers
#         self.attention.query = prune_linear_layer(self.attention.query, index)
#         self.attention.key = prune_linear_layer(self.attention.key, index)
#         self.attention.value = prune_linear_layer(self.attention.value, index)
#         self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

#         # Update hyper params and store pruned heads
#         self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
#         self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
#         self.pruned_heads = self.pruned_heads.union(heads)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#     ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#         self_outputs = self.attention(hidden_states, head_mask, output_attentions)

#         attention_output = self.output(self_outputs[0], hidden_states)

#         outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
#         return outputs


# # Copied from transformers.models.vit.modeling_vit.ViTIntermediate ViT->ViTDiffMAE
# class ViTDiffMAEIntermediate(nn.Module):
#     def __init__(self, config: ViTDiffMAEConfig) -> None:
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         if isinstance(config.hidden_act, str):
#             self.intermediate_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.intermediate_act_fn = config.hidden_act

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)

#         return hidden_states


# # Copied from transformers.models.vit.modeling_vit.ViTOutput ViT->ViTDiffMAE
# class ViTDiffMAEOutput(nn.Module):
#     def __init__(self, config: ViTDiffMAEConfig) -> None:
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)

#         hidden_states = hidden_states + input_tensor

#         return hidden_states


# # Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->ViTDiffMAE
# class ViTDiffMAELayer(nn.Module):
#     """This corresponds to the Block class in the timm implementation."""

#     def __init__(self, config: ViTDiffMAEConfig) -> None:
#         super().__init__()
#         self.chunk_size_feed_forward = config.chunk_size_feed_forward
#         self.seq_len_dim = 1
#         self.attention = ViTDiffMAEAttention(config)
#         self.intermediate = ViTDiffMAEIntermediate(config)
#         self.output = ViTDiffMAEOutput(config)
#         self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#     ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#         self_attention_outputs = self.attention(
#             self.layernorm_before(hidden_states),  # in ViTDiffMAE, layernorm is applied before self-attention
#             head_mask,
#             output_attentions=output_attentions,
#         )
#         attention_output = self_attention_outputs[0]
#         outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

#         # first residual connection
#         hidden_states = attention_output + hidden_states

#         # in ViTDiffMAE, layernorm is also applied after self-attention
#         layer_output = self.layernorm_after(hidden_states)
#         layer_output = self.intermediate(layer_output)

#         # second residual connection is done here
#         layer_output = self.output(layer_output, hidden_states)

#         outputs = (layer_output,) + outputs

#         return outputs


# # Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->ViTDiffMAE
# class ViTDiffMAEEncoder(nn.Module):
#     def __init__(self, config: ViTDiffMAEConfig) -> None:
#         super().__init__()
#         self.config = config
#         self.layer = nn.ModuleList([ViTDiffMAELayer(config) for _ in range(config.num_hidden_layers)])
#         self.gradient_checkpointing = False

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#         output_hidden_states: bool = False,
#         return_dict: bool = True,
#     ) -> Union[tuple, BaseModelOutput]:
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attentions = () if output_attentions else None

#         for i, layer_module in enumerate(self.layer):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             layer_head_mask = head_mask[i] if head_mask is not None else None

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     layer_module.__call__,
#                     hidden_states,
#                     layer_head_mask,
#                     output_attentions,
#                 )
#             else:
#                 layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

#             hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (layer_outputs[1],)

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         if not return_dict:
#             return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
#         return BaseModelOutput(
#             last_hidden_state=hidden_states,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#         )


# class ViTDiffMAEPreTrainedModel(PreTrainedModel):
#     """
#     An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
#     models.
#     """

#     config_class = ViTDiffMAEConfig
#     base_model_prefix = "vit"
#     main_input_name = "pixel_values"
#     supports_gradient_checkpointing = True

#     def _init_weights(self, module):
#         """Initialize the weights"""
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)


# VIT_MAE_START_DOCSTRING = r"""
#     This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
#     as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
#     behavior.

#     Parameters:
#         config ([`ViTDiffMAEConfig`]): Model configuration class with all the parameters of the model.
#             Initializing with a config file does not load the weights associated with the model, only the
#             configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
# """

# VIT_MAE_INPUTS_DOCSTRING = r"""
#     Args:
#         pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
#             Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
#             for details.

#         head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
#             Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

#             - 1 indicates the head is **not masked**,
#             - 0 indicates the head is **masked**.

#         output_attentions (`bool`, *optional*):
#             Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
#             tensors for more detail.
#         output_hidden_states (`bool`, *optional*):
#             Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
#             more detail.
#         return_dict (`bool`, *optional*):
#             Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# """

# @add_start_docstrings(
#     "The bare ViTDiffMAE Model transformer outputting raw hidden-states without any specific head on top.",
#     VIT_MAE_START_DOCSTRING,
# )
# class ViTDiffMAEModel(ViTDiffMAEPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config

#         self.embeddings = ViTDiffMAEEmbeddings(config)
#         self.encoder = ViTDiffMAEEncoder(config)

#         self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.embeddings.patch_embeddings

#     def _prune_heads(self, heads_to_prune):
#         """
#         Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
#         class PreTrainedModel
#         """
#         for layer, heads in heads_to_prune.items():
#             self.encoder.layer[layer].attention.prune_heads(heads)

#     @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=ViTDiffMAEModelOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         pixel_values: Optional[torch.FloatTensor] = None,   # img_claen
#         pixel_noise: Optional[torch.FloatTensor] = None,    # img_noise
#         noise: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, ViTDiffMAEModelOutput]:
#         r"""
#         Returns:

#         Examples:

#         ```python
#         >>> from transformers import AutoImageProcessor, ViTDiffMAEModel
#         >>> from PIL import Image
#         >>> import requests

#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
#         >>> model = ViTDiffMAEModel.from_pretrained("facebook/vit-mae-base")

#         >>> inputs = image_processor(images=image, return_tensors="pt")
#         >>> outputs = model(**inputs)
#         >>> last_hidden_states = outputs.last_hidden_state
#         ```"""
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if pixel_values is None:
#             raise ValueError("You have to specify pixel_values")

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         # embedding_output, mask, ids_restore = self.embeddings(pixel_values, noise=noise)
#         #################################
#         embedding_output, embedding_noise, mask, ids_restore, noise = self.embeddings(pixel_values, pixel_noise, noise=noise)
#         len_output = embedding_output.shape[1]  # visible patches
#         len_noise = embedding_noise.shape[1]    # unvisible patches (noised patches)
#         embedding_output = torch.cat([embedding_output, embedding_noise], dim=1)    # 여기가 다름. 의문임.
#         #################################

#         encoder_outputs = self.encoder(
#             embedding_output,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = encoder_outputs[0]    # last_hidden_state
#         sequence_output = self.layernorm(sequence_output)

#         ##################################
#         sequence_output, sequence_noise = torch.split(sequence_output, [len_output, len_noise], dim=1)
#         ##################################

#         if not return_dict:
#             return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

#         return ViTDiffMAEModelOutput(
#             last_hidden_state=sequence_output,
#             last_hidden_state_noise=sequence_noise,
#             mask=mask,
#             ids_restore=ids_restore,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#             noise=noise
#         )


# class ViTDiffMAEDecoder(nn.Module):
#     def __init__(self, config, num_patches):
#         super().__init__()
#         self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
#         self.decoder_pos_embed = nn.Parameter(
#             torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
#         )  # fixed sin-cos embedding

#         decoder_config = deepcopy(config)
#         decoder_config.hidden_size = config.decoder_hidden_size
#         decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
#         decoder_config.num_attention_heads = config.decoder_num_attention_heads
#         decoder_config.intermediate_size = config.decoder_intermediate_size
#         self.decoder_layers = nn.ModuleList(
#             [ViTDiffMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
#         )

#         self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
#         self.decoder_pred = nn.Linear(
#             config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
#         )  # encoder to decoder
#         self.gradient_checkpointing = False
#         self.config = config
#         self.initialize_weights(num_patches)

#         #########################
#         block_out_channels = 224
#         time_embed_dim = 224 * 4
#         flip_sin_to_cos = True
#         freq_shift = 0

#         self.time_proj = Timesteps(block_out_channels, flip_sin_to_cos, freq_shift)
#         self.time_embedding = TimestepEmbedding(block_out_channels, time_embed_dim)

#         self.time_layers = nn.ModuleList(
#             [nn.Linear(time_embed_dim, config.decoder_hidden_size, bias=True) for _ in range(config.decoder_num_hidden_layers)]
#         )
#         #########################


#     def initialize_weights(self, num_patches):
#         # initialize (and freeze) position embeddings by sin-cos embedding
#         decoder_pos_embed = get_2d_sincos_pos_embed(
#             self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
#         )
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

#     def forward(
#         self,
#         hidden_states,          # encoder output of visible patches
#         hidden_states_noise,    # encoder output of unvisible patches (nosied)  # 의문
#         timesteps,
#         ids_restore,
#         output_attentions=False,
#         output_hidden_states=False,
#         return_dict=True,
#     ):

#         ######################
#         t_emb = self.time_proj(timesteps)
#         emb = self.time_embedding(t_emb)
#         ######################

#         # embed tokens
#         x = self.decoder_embed(hidden_states)

#         # # append mask tokens to sequence
#         # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
#         # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
#         # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
#         # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

#         ##################
#         mask_tokens = self.decoder_embed(hidden_states_noise)
#         x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
#         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
#         x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
#         ##################

#         # add pos embed
#         hidden_states = x + self.decoder_pos_embed

#         # apply Transformer layers (blocks)
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attentions = () if output_attentions else None
#         for i, (layer_module, time_layer_module) in enumerate(zip(self.decoder_layers, self.time_layers)):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     layer_module.__call__,
#                     hidden_states,
#                     None,
#                     output_attentions,
#                 )
#             else:
#                 #######################
#                 layer_time = time_layer_module(emb)
#                 layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)
#                 layer_outputs = (layer_outputs[0] + layer_time.unsqueeze(1), )
#                 ########################

#             hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (layer_outputs[1],)

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         hidden_states = self.decoder_norm(hidden_states)

#         # predictor projection
#         logits = self.decoder_pred(hidden_states)

#         # remove cls token
#         logits = logits[:, 1:, :]

#         if not return_dict:
#             return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
#         return ViTDiffMAEDecoderOutput(
#             logits=logits,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#         )


# @add_start_docstrings(
#     """The ViTDiffMAE Model transformer with the decoder on top for self-supervised pre-training.

#     <Tip>

#     Note that we provide a script to pre-train this model on custom data in our [examples
#     directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

#     </Tip>

#     """,
#     VIT_MAE_START_DOCSTRING,
# )
# class ViTDiffMAEForPreTraining(ViTDiffMAEPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config

#         self.vit = ViTDiffMAEModel(config)
#         self.decoder = ViTDiffMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)

#         # ###################################
#         # flip_sin_to_cos = True
#         # freq_shift = 0
#         # block_out_channels = [224, 448, 672, 896]
#         # timestep_input_dim = block_out_channels[0]
#         # time_embed_dim = block_out_channels[0] * 4
#         # self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
#         # self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
#         # ###################################

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.vit.embeddings.patch_embeddings

#     def _prune_heads(self, heads_to_prune):
#         """
#         Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
#         class PreTrainedModel
#         """
#         for layer, heads in heads_to_prune.items():
#             self.encoder.layer[layer].attention.prune_heads(heads)

#     def patchify(self, pixel_values):
#         """
#         Args:
#             pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
#                 Pixel values.

#         Returns:
#             `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
#                 Patchified pixel values.
#         """
#         patch_size, num_channels = self.config.patch_size, self.config.num_channels
#         # sanity checks
#         if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
#             raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
#         if pixel_values.shape[1] != num_channels:
#             raise ValueError(
#                 "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
#             )

#         # patchify
#         batch_size = pixel_values.shape[0]
#         num_patches_one_direction = pixel_values.shape[2] // patch_size
#         patchified_pixel_values = pixel_values.reshape(
#             batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size
#         )
#         patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
#         patchified_pixel_values = patchified_pixel_values.reshape(
#             batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels
#         )
#         return patchified_pixel_values

#     def unpatchify(self, patchified_pixel_values):
#         """
#         Args:
#             patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
#                 Patchified pixel values.

#         Returns:
#             `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
#                 Pixel values.
#         """
#         patch_size, num_channels = self.config.patch_size, self.config.num_channels
#         num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
#         # sanity check
#         if num_patches_one_direction**2 != patchified_pixel_values.shape[1]:
#             raise ValueError("Make sure that the number of patches can be squared")

#         # unpatchify
#         batch_size = patchified_pixel_values.shape[0]
#         patchified_pixel_values = patchified_pixel_values.reshape(
#             batch_size,
#             num_patches_one_direction,
#             num_patches_one_direction,
#             patch_size,
#             patch_size,
#             num_channels,
#         )
#         patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
#         pixel_values = patchified_pixel_values.reshape(
#             batch_size,
#             num_channels,
#             num_patches_one_direction * patch_size,
#             num_patches_one_direction * patch_size,
#         )
#         return pixel_values

#     def forward_loss(self, pixel_values, pred, mask):
#         """
#         Args:
#             pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
#                 Pixel values.
#             pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
#                 Predicted pixel values.
#             mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
#                 Tensor indicating which patches are masked (1) and which are not (0).

#         Returns:
#             `torch.FloatTensor`: Pixel reconstruction loss.
#         """
#         target = self.patchify(pixel_values)

#         if self.config.norm_pix_loss:
#             mean = target.mean(dim=-1, keepdim=True)
#             var = target.var(dim=-1, keepdim=True)
#             target = (target - mean) / (var + 1.0e-6) ** 0.5

#         loss = (pred - target) ** 2 # pred: [batch_size, num_patches, patch_size**2 * num_channels].
#         loss = loss.mean(dim=-1)  # [N, L], # [batch_size, num_patches] mean loss per patch # L2 Loss

#         loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
#         return loss


#     @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=ViTDiffMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         pixel_values: Optional[torch.FloatTensor] = None,
#         pixel_noise: Optional[torch.FloatTensor] = None,
#         timesteps: Optional[torch.FloatTensor] = None,
#         baldnessmap: Optional[torch.FloatTensor] = None,
#         noise: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         added_noise: Optional[torch.FloatTensor] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, ViTDiffMAEForPreTrainingOutput]:
#         r"""
#         Returns:

#         Examples:

#         ```python
#         >>> from transformers import AutoImageProcessor, ViTDiffMAEForPreTraining
#         >>> from PIL import Image
#         >>> import requests

#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
#         >>> model = ViTDiffMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

#         >>> inputs = image_processor(images=image, return_tensors="pt")
#         >>> outputs = model(**inputs)
#         >>> loss = outputs.loss
#         >>> mask = outputs.mask
#         >>> ids_restore = outputs.ids_restore
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.vit(
#             pixel_values,   # img_clean
#             pixel_noise,    # img_noise
#             noise=noise,                                # None
#             head_mask=head_mask,                        # None
#             output_attentions=output_attentions,        # None
#             output_hidden_states=output_hidden_states,  # None
#             return_dict=return_dict,                    # None
#         )

#         latent = outputs.last_hidden_state              # encoding output of visible patches
#         latent_noise = outputs.last_hidden_state_noise  # encoding output of unvisible patches (noised)
#         ids_restore = outputs.ids_restore               # patches index for visible/unvisible patches
#         mask = outputs.mask                             # mask patches (0: visible, 1: unvisible)
#         noise = outputs.noise

#         ###############
#         # latent = outputs.last_hidden_state_noise
#         ###############

#         ###############################
#         timesteps = timesteps * torch.ones(pixel_values.shape[0], dtype=timesteps.dtype, device=timesteps.device)
#         ###############################

#         decoder_outputs = self.decoder(latent, latent_noise, timesteps, ids_restore)
#         logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        
#         # print("pixel_values.shape", pixel_values.shape) # torch.Size([64, 64, 256, 256])
#         # print("baldnessmap.shape", baldnessmap.shape)   # torch.Size([64, 64, 256, 256])

#         loss = self.forward_loss(pixel_values, logits, mask)
#         # loss = self.forward_loss(pixel_values * baldnessmap, logits, mask)

#         if not return_dict:
#             output = (logits, mask, ids_restore) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return ViTDiffMAEForPreTrainingOutput(
#             loss=loss,
#             logits=logits,
#             mask=mask,
#             ids_restore=ids_restore,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             noise=outputs.noise,
#         )