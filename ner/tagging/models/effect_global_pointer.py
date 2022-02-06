#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：global_pointer.py
@Author  ：hbx
@Date    ：2022/2/5 13:30 
'''

import torch
from overrides import overrides
from typing import Dict, List, Optional
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from tagging.nn.multilabel_cross_entropy import MultilabelCE
from tagging.metrics.globa_pointer_f1_measure import GolbalPointerMeasure
import json


class SinusoidalPositionEmbedding(torch.nn.Module):
    """定义Sin-Cos位置Embedding
    """

    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


@Model.register('effect_global_pointer')
class EffectGlobalPointer(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 entity_id_path: str,
                 label_num: int,
                 rope: bool = True,
                 inner_dim: int = 64,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 feedforward: Optional[FeedForward] = None,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs):
        super(EffectGlobalPointer, self).__init__(vocab, **kwargs)
        self._label_num = label_num
        self._rope = rope
        self._inner_dim = inner_dim
        self._embedder = embedder
        self._encoder = encoder
        self._hidden_dim = self._embedder.get_output_dim()
        if self._encoder:
            self._hidden_dim = self._encoder.get_output_dim()
        self._feedforward = feedforward
        self._dense_1 = torch.nn.Linear(self._hidden_dim, self._inner_dim * 2)
        self._dense_2 = torch.nn.Linear(self._inner_dim * 2, self._label_num * 2)
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._criterion = MultilabelCE()

        with open(entity_id_path, mode='r', encoding='utf8') as f:
            self._entity_id_dict = json.load(f)

        self._f1 = GolbalPointerMeasure(self._entity_id_dict)
        initializer(self)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._f1.get_metric(reset)

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, tokens: Dict[str, Dict[str, torch.Tensor]], labels=None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        seq_out = self._embedder(tokens)
        if self._encoder:
            seq_out = self._encoder(seq_out, mask)

        if self._dropout:
            seq_out = self._dropout(seq_out)

        batch_size = seq_out.shape[0]
        seq_length = seq_out.shape[1]

        # (batch_size, seq_len, inner_dim*2)
        qw_kw = self._dense_1(seq_out)
        # qw kw :(batch_size, seq_len, inner_dim)
        qw, kw = qw_kw[..., ::2], qw_kw[..., 1::2]

        if self._rope:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = SinusoidalPositionEmbedding(self._inner_dim, 'zero')(qw_kw)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # (batch_size, seq_len, seq_len)
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self._inner_dim ** 0.5
        # (batch_size, 2 * entity_type_size, seq)
        bias = torch.einsum('bnh->bhn', self._dense_2(qw_kw)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits[:, None] 增加一个维度
        # (batch_size, entity_type_size, seq_len, seq_len)
        logits = self.add_mask_tril(logits, mask=mask)
        output_dict = {
            'logits': logits
        }

        if labels is not None:
            output_dict['loss'] = self._criterion(logits, labels)
            self._f1(logits, labels)

        return output_dict
