#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：biaffine.py
@Author  ：hbx
@Date    ：2022/2/10 11:17 
'''
import torch
from overrides import overrides
from typing import Dict, Optional
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from tagging.nn.layer.biaffine_layer import BiaffineLayer
import json


@Model.register('biaffine')
class Biaffine(Model):

    def __init__(self, vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 entity_id_path: str,
                 label_num: int,
                 start_encoder: FeedForward,
                 end_encoder: FeedForward,
                 biaffine_dim: int = 128,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs
                 ):
        super(Biaffine, self).__init__(vocab, **kwargs)
        self.__embedder = embedder
        self.__biaffine_dim = biaffine_dim
        self.__label_num = label_num
        with open(entity_id_path, mode='r', encoding='utf8') as f:
            self.__entity_id_dict = json.load(f)

        self.__encoder = encoder
        self.__start_encoder = start_encoder
        self.__end_encoder = end_encoder

        self.__biaffine_layer = BiaffineLayer(biaffine_dim, label_num)

        if dropout:
            self.__dropout = torch.nn.Dropout(dropout)
        else:
            self.__dropout = None

        self.__criterion = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass

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
        seq_out = self.__embedder(tokens)

        if self.__encoder:
            seq_out = self.__encoder(seq_out, mask)
        if self.__dropout:
            seq_out = self.__dropout(seq_out)
        batch_size = seq_out.shape[0]
        seq_length = seq_out.shape[1]

        start_out = self.__start_encoder(seq_out)
        end_out = self.__end_encoder(seq_out)

        logits = self.__biaffine_layer(start_out, end_out)
        # padding mask
        pad_mask = mask.unsqueeze(1).unsqueeze(-1).expand(mask.shape[0], seq_length, seq_length, self.__label_num)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        # 排除下三角
        pad_mask = torch.tril(pad_mask, -1)
        logits = logits * pad_mask.float() - (1 - pad_mask.float()) * 1e12

        # 排除下三角
        # mask = torch.tril(torch.ones_like(logits), -1)
        # logits = logits - mask * 1e12


        output_dict = {
            'logits': logits
        }

        if labels is not None:

            output_dict['loss'] = self.__criterion(logits.contiguous().view(size=(-1, self.__label_num)), labels.contiguous().view(size=(-1,)))

        return output_dict
