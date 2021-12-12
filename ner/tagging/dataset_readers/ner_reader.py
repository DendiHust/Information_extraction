'''
 # @ Author: xuan
 # @ Create Time: 2021-08-23 17:34:55
 # @ Modified by: xuan
 # @ Modified time: 2021-08-23 17:35:02
 # @ Description:
 '''
import logging
from overrides import overrides
from typing import Dict, List, Optional, Iterator
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from tagging.utils.data_util import convert_2_crf_example
import json

logger = logging.getLogger(__name__)


@DatasetReader.register("emr_ner")
class NERReader(DatasetReader):

    def __init__(self,
                 max_length: int = 512,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_length = max_length
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer(lowercase_tokens=True)}

    @overrides
    def _read(self, file_path) -> Iterator[Instance]:
        with open(file_path, mode='r', encoding='utf8') as f:
            tmp_data = json.load(f)
        for item in tmp_data:
            tmp = convert_2_crf_example(item)
            text = tmp['text']
            labels = tmp['labels']
            yield self.text_to_instance(text, labels)
    
    @overrides
    def text_to_instance(self, words: List[str], ner_tags: List[str] = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields['tokens'] = tokens
        if ner_tags:
            fields['labels'] = SequenceLabelField(ner_tags, tokens)

        return Instance(fields)


if __name__ == '__main__':

    ner_reader = NERReader()
    for item in list(ner_reader._read('./new_data/rel_data_ner_val_data_tag.json')):
        print(item)
        break
