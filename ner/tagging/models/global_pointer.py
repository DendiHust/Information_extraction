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

@Model.register('global_pointer')
class GlobalPointer(Model):
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
        super(GlobalPointer, self).__init__(vocab, **kwargs)
        self._label_num = label_num
        self._rope = rope
        self._inner_dim = inner_dim
        self._embedder = embedder
        self._encoder = encoder
        self._hidden_dim = self._embedder.get_output_dim()
        if self._encoder:
            self._hidden_dim = self._encoder.get_output_dim()
        self._feedforward = feedforward
        self._dense = torch.nn.Linear(self._hidden_dim, self._label_num * self._inner_dim * 2)
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

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        # embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, tokens: Dict[str, Dict[str, torch.Tensor]], labels=None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        seq_out = self._embedder(tokens)
        if self._encoder:
            seq_out = self._encoder(seq_out, mask)

        if self._dropout:
            seq_out = self._dropout(seq_out)

        batch_size = seq_out.shape[0]
        seq_length = seq_out.shape[1]

        # (batch_size, seq_len, ent_type_size*inner_dim*2)
        qw_kw = self._dense(seq_out)
        qw_kw = torch.split(qw_kw, self._inner_dim * 2, dim=-1)
        # (batch_size, seq_len, ent_type_size, inner_dim*2)
        qw_kw = torch.stack(qw_kw, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = qw_kw[..., :self._inner_dim], qw_kw[..., self._inner_dim:]

        if self._rope:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_length, self._inner_dim).to(seq_out.device)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self._label_num, seq_length, seq_length)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask.float() - (1 - pad_mask.float()) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        output_dict = {
            'logits': logits
        }
        if labels is not None:
            output_dict['loss'] = self._criterion(logits, labels)
            self._f1(logits, labels)

        return output_dict


