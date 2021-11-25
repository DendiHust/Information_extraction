'''
Author: hbx
Date: 2021-11-24 17:32:41
LastEditTime: 2021-11-25 11:18:08
LastEditors: Please set LastEditors
FilePath: \informatin_extraction\ner\tagging\dataset_readers\data_reader.py
'''
import logging
from allennlp.common.registrable import Registrable
from allennlp.data.token_indexers import token_indexer
from overrides import overrides
from typing import Dict, List, Iterator,Iterable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer

import json
logger = logging.getLogger(__name__)

@DatasetReader.register("tag_reader")
class TagReader(DatasetReader):
    def __init__(self,
                 max_length: int = 512,
                 model_type = 'crf',
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_length = max_length
        self._model_type = model_type
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer(lowercase_tokens=True)
        }
        


    
    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        with open(file_path, mode='r', encoding='utf8') as f:
            tmp_data = json.load(f)
        
        for item in tmp_data:
            text = item['text']
            labels = item['labels']



    @overrides
    def text_to_instance(self, *inputs) -> Instance:
        return super().text_to_instance(*inputs)



if __name__ == '__main__':
    pass