#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：bert_crf.py
@Author  ：hbx
@Date    ：2021/12/6 16:33 
'''
import torch
from overrides import overrides
from typing import Dict, List
from allennlp.data.vocabulary import Vocabulary
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.nn.util import get_text_field_mask



@Model.register('bert_crf')
class BertCrf(Model):
    def __init__(self, vocab: Vocabulary,
                 bert_path: str,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs):
        super().__init__(vocab, **kwargs)
        self._bert_model = BertModel.from_pretrained(bert_path)
        self._bert_config = BertConfig.from_pretrained(bert_path)
        self._bert_config.num_labels = vocab.get_vocab_size('labels')

        initializer(self)


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        pass


