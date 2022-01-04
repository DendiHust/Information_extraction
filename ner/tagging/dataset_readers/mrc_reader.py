#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：mrc_reader.py
@Author  ：hbx
@Date    ：2022/1/4 15:47 
'''
import logging
from overrides import overrides
from typing import Dict, List,Optional,Iterable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer

import json

logger = logging.getLogger(__name__)

@DatasetReader.register("mrc_reader")
class MRCReader(DatasetReader):

    def __init__(self,
                 pretrained_model_path,
                 max_length: int=512,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_length = max_length
        self._pretrained_model_path = pretrained_model_path
        self._token_indexers = {'tokens': PretrainedTransformerIndexer(self._pretrained_model_path)}

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        pass

