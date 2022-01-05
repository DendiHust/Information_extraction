#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：mrc.py
@Author  ：hbx
@Date    ：2022/1/5 11:24 
'''
import torch
from overrides import overrides
from typing import Dict, List, Optional
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.util import get_text_field_mask
from torch import nn

import torch.nn.functional as F

@Model.register('mrc')
class MRC(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 start_feedforward: FeedForward,
                 end_feedforward: FeedForward,
                 dropout: float = 0.3,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs):
        super().__init__(vocab, **kwargs)

        self._embedder = embedder

        if dropout is not None:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        _hidden_dim = self._embedder.get_output_dim()
        self._start_feedforward = start_feedforward
        self._end_feedforward = end_feedforward

        self._criterion = nn.CrossEntropyLoss()

        initializer(self)


    def forward(self, question_with_context: Dict[str, Dict[str, torch.Tensor]], start_labels = None, end_labels = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(question_with_context)
        embedded = self._embedder(question_with_context)

        start_logits = self._start_feedforward(embedded)
        end_logits = self._end_feedforward(embedded)
        output_dict = {
            "start_logits": start_logits,
            "end_logits": end_logits
        }

        active_loss = question_with_context['tokens']['type_ids'] == 1
        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]

        if start_labels is not None:
            active_start_labels = start_labels[active_loss]
            active_end_labels = end_labels[active_loss]
            start_loss = self._criterion(active_start_logits, active_start_labels)
            end_loss = self._criterion(active_end_logits, active_end_labels)
            output_dict['loss'] = start_loss + end_loss


        return output_dict






