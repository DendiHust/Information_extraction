#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：global_pointer_reader.py
@Author  ：hbx
@Date    ：2022/2/5 16:40 
'''
import torch
from overrides import overrides
from typing import Dict, List, Iterable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
import json
import numpy as np


@DatasetReader.register('global_pointer_reader')
class GlobalPointerReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 max_length: int = 256,
                 **kwargs
                 ):
        super(GlobalPointerReader, self).__init__(**kwargs)
        self.__max_length = max_length
        self.__token_indexers = token_indexers
        self.__user_pretrained_model = True
        if not isinstance(self.__token_indexers['tokens'], PretrainedTransformerIndexer):
            self.__user_pretrained_model = False

    def convert_2_global_pointer_example(self, data_item: Dict, entity_id_dict: Dict = None) -> Dict:
        if entity_id_dict is None:
            with open('./data/mid_data/global_ent2id.json', mode='r', encoding='utf8') as f:
                entity_id_dict = json.load(f)
        entity_type_number = len(entity_id_dict)
        text = data_item['text']

        if self.__user_pretrained_model:
            text = text[: self.__max_length - 2]
            seq_length = len(text)
            # [cls] text [seq]
            labels = np.zeros((entity_type_number, seq_length + 2, seq_length + 2), dtype=np.long)
            for entity_item in data_item['labels']:
                if entity_item[3] >= self.__max_length - 2:
                    continue
                labels[entity_id_dict[entity_item[1]], entity_item[2] + 1, entity_item[3]] = 1
        else:
            text = text[: self.__max_length]
            seq_length = len(text)
            labels = np.zeros((entity_type_number, seq_length, seq_length), dtype=np.long)
            for entity_item in data_item['labels']:
                if entity_item[3] - 1 >= self.__max_length:
                    continue
                labels[entity_id_dict[entity_item[1]], entity_item[2], entity_item[3] - 1] = 1
        return {'text': text, 'labels': labels}

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        with open(file_path, mode='r', encoding='utf8') as f:
            data = json.load(f)

        for item in data:
            tmp_item = self.convert_2_global_pointer_example(item)
            context = tmp_item['text']
            labels = tmp_item['labels']
            yield self.text_to_instance(list(context), labels)

    @overrides
    def text_to_instance(self, tokens: List[str], labels: List = None) -> Instance:
        fields: Dict[str, Field] = {}

        if self.__user_pretrained_model:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
        tokens = TextField([Token(w) for w in tokens], self.__token_indexers)
        fields['tokens'] = tokens
        if labels is not None:
            fields['labels'] = ArrayField(torch.from_numpy(labels))

        return Instance(fields)
