#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：mrc_reader.py
@Author  ：hbx
@Date    ：2022/1/4 15:47 
'''
import logging

import torch
from overrides import overrides
from typing import Dict, List, Optional, Iterable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from tagging.utils.data_util import convert_2_mrc_exapmple
import numpy as np

import json

logger = logging.getLogger(__name__)

@DatasetReader.register("mrc_reader")
class MRCReader(DatasetReader):

    def __init__(self,
                 pretrained_model_path,
                 max_length: int = 512,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_length = max_length
        self._pretrained_model_path = pretrained_model_path
        self._token_indexers = {'tokens': PretrainedTransformerIndexer(self._pretrained_model_path)}
        self._tokenizer = PretrainedTransformerTokenizer(self._pretrained_model_path)

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        with open(file_path, mode='r', encoding='utf8') as f:
            tmp_data = json.load(f)

        for item in tmp_data:
            tmp_item = convert_2_mrc_exapmple(item)
            context = tmp_item['text']
            for _labels in tmp_item['labels']:
                question = _labels['question']
                question_type = _labels['question_type']
                start_labels = _labels['start_labels']
                end_labels = _labels['end_labels']
                yield self.text_to_instance(context, question, start_labels, end_labels, question_type)


    @overrides
    def text_to_instance(
            self,
            context: List[str],
            question: List[str],
            start_labels: List[int] = None,
            end_labels: List[int] = None,
            question_type: str = None
    ) -> Instance:
        fields: Dict[str, Field] = {}
        _limit_length = self._max_length - len(question) - 3

        if len(context) > _limit_length:
            context = context[:_limit_length]
            if start_labels is not None:
                start_labels = start_labels[:_limit_length]
                end_labels = end_labels[:_limit_length]

        if start_labels is not None:
            start_labels = [0] + [0] * len(question) + [0] + start_labels + [0]
            end_labels = [0] + [0] * len(question) + [0] + end_labels + [0]

        question_tokens = [Token(w) for w in question]
        context_tokens = [Token(w) for w in context]

        tokens = TextField(self._tokenizer.add_special_tokens(question_tokens, context_tokens))

        fields['question_with_context'] = tokens

        metadata = {
            'question': question,
            'question_type': question_type,
            'context': context
        }
        fields['metadata'] = MetadataField(metadata)

        if start_labels is not None:
            fields['start_labels'] = ArrayField(torch.from_numpy(np.array(start_labels)).long())
            fields['end_labels'] = ArrayField(torch.from_numpy(np.array(end_labels)).long())



        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["question_with_context"].token_indexers = self._token_indexers