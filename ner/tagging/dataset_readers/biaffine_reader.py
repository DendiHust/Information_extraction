#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：biaffine_reader.py
@Author  ：hbx
@Date    ：2022/2/11 10:21 
'''
import torch
from overrides import overrides
from typing import Dict, List, Iterable
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
import json
import numpy as np


@DatasetReader.register('biaffine_reader')
class BiaffineReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 max_length: int = 256,
                 **kwargs
                 ):
        super(BiaffineReader, self).__init__(**kwargs)
        self.__max_length = max_length
        self.__token_indexers = token_indexers
        self.__user_pretrained_model = True
        if not isinstance(self.__token_indexers['tokens'], PretrainedTransformerIndexer):
            self.__user_pretrained_model = False

    def convert_2_biaffine_example(self, data_item: Dict, entity_id_dict: Dict = None) -> Dict:
        if entity_id_dict is None:
            with open('./data/mid_data/biaffine_ent2id.json', mode='r', encoding='utf8') as f:
                entity_id_dict = json.load(f)
        text = data_item['text']

        if self.__user_pretrained_model:
            text = text[: self.__max_length - 2]
            seq_length = len(text)
            # [cls] text [seq]
            labels = np.zeros((seq_length + 2, seq_length + 2), dtype=np.long)
            # 实体的结束位置大于文本最大值则抛弃该实体
            for entity_item in data_item['labels']:
                if entity_item[3] >= self.__max_length - 2:
                    continue
                labels[entity_item[2] + 1, entity_item[3]] = entity_id_dict[entity_item[1]]
        else:
            text = text[: self.__max_length]
            seq_length = len(text)
            labels = np.zeros((seq_length, seq_length), dtype=np.long)
            # 实体的结束位置大于文本最大值则抛弃该实体
            for entity_item in data_item['labels']:
                if entity_item[3] - 1 >= self.__max_length:
                    continue
                labels[entity_item[2], entity_item[3] - 1] = entity_id_dict[entity_item[1]]
        return {'text': text, 'labels': labels}

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        with open(file_path, mode='r', encoding='utf8') as f:
            data = json.load(f)

        for item in data:
            tmp_item = self.convert_2_biaffine_example(item)
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
            fields['labels'] = ArrayField(torch.from_numpy(labels).long())

        return Instance(fields)
