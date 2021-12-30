#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：tag_reader.py
@Author  ：hbx
@Date    ：2021/12/6 15:26 
'''

import logging
from overrides import overrides
from typing import Dict, List, Optional, Iterable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer

from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
# from transformers.models.bert.tokenization_bert import BertTokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from tagging.utils.data_util import convert_2_tagger_example

import json

logger = logging.getLogger(__name__)

@DatasetReader.register("tagger_reader")
class TagReader(DatasetReader):
    def __init__(self,
                 max_length: int = 512,
                 pretrained_model_path: Optional[str] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_length = max_length
        if pretrained_model_path is not None:
            self._pretrained_model_path = pretrained_model_path
            self._token_indexers = {'tokens': PretrainedTransformerIndexer(self._pretrained_model_path)}
        else:
            self._pretrained_model_path = None
            self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(lowercase_tokens=True)}


    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        with open(file_path, mode='r', encoding='utf8') as f:
            tmp_data = json.load(f)

        for item in tmp_data:
            tmp = convert_2_tagger_example(item)
            text = tmp['text']
            labels = tmp['labels']
            yield self.text_to_instance(list(text), labels)

    @overrides
    def text_to_instance(self, tokens: List[str], labels: List[str] = None) -> Instance:
        fields: Dict[str, Field] = {}
        if self._pretrained_model_path:
            if len(tokens) > self._max_length - 2:
                tokens = tokens[:self._max_length - 2]
                labels = labels[:self._max_length - 2]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            if labels:
                labels = ["O"] + labels + ["O"]

        tokens = TextField([Token(w) for w in tokens], self._token_indexers)
        fields['tokens'] = tokens
        if labels:
            fields['labels'] = SequenceLabelField(labels, tokens)
        return Instance(fields)


if __name__ == '__main__':
    # test_reader = TagReader()
    # dataset = list(test_reader.read('../../data/raw_data/tmp.json'))[0]
    # for token, label in zip(dataset['tokens'], dataset['labels']):
    #     print(f'{token}\t{label}')
    from transformers.models.bert.tokenization_bert import BertTokenizer

    tmp = BertTokenizer.from_pretrained('../../bert-base-chinese')
    print(tmp.convert_tokens_to_ids('我'))
