#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：data_reader.py
@Author  ：hbx
@Date    ：2021/12/6 15:26 
'''

import logging
from allennlp.common.registrable import Registrable
from allennlp.data.token_indexers import token_indexer
from overrides import overrides
from typing import Dict, List, Iterator, Iterable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers.pretrained_transformer_mismatched_indexer import \
    PretrainedTransformerMismatchedIndexer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
# from transformers.models.bert.tokenization_bert import BertTokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from tagging.utils.data_util import convert_2_crf_example
from transformers.models.bert.tokenization_bert import  BertTokenizer


from transformers.models.bert.tokenization_bert import BertTokenizer

import json

logger = logging.getLogger(__name__)


@DatasetReader.register("tag_reader")
class TagReader(DatasetReader):
    def __init__(self,
                 max_length: int = 512,
                 model_type='crf',
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_length = max_length
        self._model_type = model_type
        self._token_indexers = token_indexers or {
            # 'tokens': PretrainedTransformerMismatchedIndexer('bert-base-chinese')
            # 'tokens': PretrainedTransformerMismatchedIndexer('bert-base-chinese')
            'tokens': PretrainedTransformerIndexer('./bert-base-chinese')
        }
        self._bert_tokenizer =BertTokenizer.from_pretrained('./bert-base-chinese')

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        with open(file_path, mode='r', encoding='utf8') as f:
            tmp_data = json.load(f)

        for item in tmp_data:
            tmp = convert_2_crf_example(item)
            text = tmp['text']
            labels = tmp['labels']
            yield self.text_to_instance(list(text), labels)

    @overrides
    def text_to_instance(self, tokens: List[str], labels: List[str] == None) -> Instance:
        fields: Dict[str, Field] = {}
        if len(tokens) > self._max_length - 2:
            tokens = tokens[:self._max_length - 2]
            labels = labels[:self._max_length - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        labels = ["O"] + labels + ["O"]
        tokens = TextField([Token(w, text_id=self._bert_tokenizer.convert_tokens_to_ids(w)) for w in tokens], self._token_indexers)
        fields['tokens'] = tokens
        if labels:
            fields['labels'] = SequenceLabelField(labels, tokens)
        return Instance(fields)


if __name__ == '__main__':
    # test_reader = TagReader()
    # dataset = list(test_reader.read('../../data/raw_data/tmp.json'))[0]
    # for token, label in zip(dataset['tokens'], dataset['labels']):
    #     print(f'{token}\t{label}')
    tmp = BertTokenizer.from_pretrained('../..//bert-base-chinese')
    print(tmp.convert_tokens_to_ids('我'))