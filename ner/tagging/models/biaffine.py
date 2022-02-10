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


        initializer(self)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        pass