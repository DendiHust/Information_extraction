'''
 # @ Author: xuan
 # @ Create Time: 2021-08-27 10:45:11
 # @ Modified by: xuan
 # @ Modified time: 2021-08-27 10:45:18
 # @ Description:
 '''
import logging
from allennlp.data.fields.array_field import ArrayField
from overrides import overrides
from typing import Dict, List, Optional, Iterator, Tuple, Set
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import numpy as np

import json



@DatasetReader.register("relation_base_reader")
class ReReader(DatasetReader):

    def __init__(self,
                 config_file_path: str,
                 max_length: int = 512,
                 negative_sample_number: int = 10,
                 use_entity_tag = True,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._negative_sample_number = negative_sample_number

        with open(config_file_path, mode='r', encoding='utf8') as f:
            rel_info_dict = json.load(f)
        # 关系标签信息
        self._rel_tag_info_dict = rel_info_dict['rel_tag_info']
        self._rel_tag_info_dict_rev = {
            v: k for k, v in self._rel_tag_info_dict.items()}
        # 关系label 与 index 信息
        self._rel_label_2_id = rel_info_dict['rel_label_2_id']
        self._rel_id_2_label = rel_info_dict['rel_id_2_label']
        # 实体标签信息  type: type_code
        self._ner_tag_info_dict = rel_info_dict['ner_tag_info']
        # 实体标签信息  type_code: type
        self._ner_tag_info_dict_rev = {
            v: k for k, v in self._ner_tag_info_dict.items()}
        # 实体 label 与 index 信息
        self._ner_tag_2_id = rel_info_dict['ner_tag_2_id']
        self._ner_id_2_tag = rel_info_dict['ner_id_2_tag']

        self._max_length = max_length
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer(lowercase_tokens=True)}
        self._use_entity_tag = use_entity_tag




    def __negative_sampling(self, info_dict: Dict, negative_sampling_number: int = 10) -> Dict:
        # (entity_start_pos, entity_end_pos, entity_type, entity_text)
        entity_sets: Set[Tuple[int, int, str, str]] = set()
        for item in info_dict['ner_list']:
            entity_sets.add(
                (item['entity_start_pos'], item['entity_end_pos'],
                 item["entity_type"], item['entity_text']))

        pos_rel_dict: Dict[Tuple[int, int, str, str],
                           Set[Tuple[int, int, str, str]]] = {}
        # 统计正样本
        for item in info_dict['relation_list']:
            tmp_key = (item['start_entity_start_pos'], item['start_entity_end_pos'],
                       item["start_entity_type"], item['start_entity_text'])
            tmp_val = (item['end_entity_start_pos'], item['end_entity_end_pos'],
                       item["end_entity_type"], item['end_entity_text'])
            if tmp_key not in pos_rel_dict:
                pos_rel_dict[tmp_key] = set()
            pos_rel_dict[tmp_key].add(tmp_val)

        neg_sample_result: List[Dict[str, str]] = list()

        # 负采样
        for k, v in pos_rel_dict.items():
            sample_list = list(entity_sets - v - set([k]))

            if len(sample_list) > negative_sampling_number:
                np.random.shuffle(sample_list)

            limit_index = min(negative_sampling_number, len(sample_list))

            for tmp_v in sample_list[:limit_index]:
                # 如果两个实体之间的距离超过 30就跳过
                if abs(tmp_v[0] - k[0]) > 30:
                    continue
                neg_sample_result.append({
                    'start_entity_start_pos': k[0],
                    'start_entity_end_pos': k[1],
                    'start_entity_type': k[2],
                    'start_entity_text': k[3],
                    'relation_type': 'None',
                    'end_entity_start_pos': tmp_v[0],
                    'end_entity_end_pos': tmp_v[1],
                    'end_entity_type': tmp_v[2],
                    'end_entity_text': tmp_v[3],
                })
        info_dict["relation_list"].extend(neg_sample_result)
        return info_dict

    def __get_sequence_entity_tags(self, start_entity_pos: Tuple[int, int],
                                   start_entity_type: int,
                                   end_entity_pos: Tuple[int, int],
                                   end_entity_type: int,
                                   seq_length: int = 512) -> List[int]:

        seq_entity_tags = [0] * seq_length
        for i in range(start_entity_pos[0], start_entity_pos[1]):
            seq_entity_tags[i] = start_entity_type

        for i in range(end_entity_pos[0], end_entity_pos[1]):
            seq_entity_tags[i] = end_entity_type
        return seq_entity_tags

    @overrides
    def _read(self, file_path) -> Iterator[Instance]:
        with open(file_path, mode='r', encoding='utf8') as f:
            tmp_data = json.load(f)

        for item in tmp_data:
            item = self.__negative_sampling(item, self._negative_sample_number)
            tokens = list(item['text'][:self._max_length])
            for rel_item in item['relation_list']:
                if rel_item['end_entity_end_pos'] >= self._max_length or rel_item['start_entity_end_pos'] >= self._max_length:
                    continue

                start_entity_pos = (
                    rel_item['start_entity_start_pos'], rel_item['start_entity_end_pos'])
                start_entity_type = rel_item['start_entity_type']
                end_entity_pos = (
                    rel_item['end_entity_start_pos'], rel_item['end_entity_end_pos'])
                end_entity_type = rel_item['end_entity_type']
                labels = 'None' if rel_item['relation_type'] == 'None' else self._rel_tag_info_dict[rel_item['relation_type']]
                yield self.text_to_instance(tokens, start_entity_pos, start_entity_type, end_entity_pos, end_entity_type, labels)

    @overrides
    def text_to_instance(self, tokens: List[str],
                         start_entity_pos: Tuple[int, int],
                         start_entity_type: str,
                         end_entity_pos: Tuple[int, int],
                         end_entity_type: str,
                         labels: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = TextField([Token(w) for w in tokens], self._token_indexers)
        fields['tokens'] = tokens

        start_entity_type_id = self._ner_tag_info_dict[start_entity_type]
        end_entity_type_id = self._ner_tag_info_dict[end_entity_type]

        seq_entity_tags = self.__get_sequence_entity_tags(
            start_entity_pos, start_entity_type_id, end_entity_pos, end_entity_type_id, len(tokens))
        fields['seq_entity_tags'] = SequenceLabelField(
            seq_entity_tags, tokens, label_namespace='seq_entity_tags')

        if labels:
            fields['labels'] = LabelField(labels)

        return Instance(fields)


if __name__ == '__main__':
    re_reader = ReReader('./conf/rel_label_info.json')
    info_dict = {
        "ner_list": [
            {
                "entity_end_pos": 6,
                "entity_start_pos": 2,
                "entity_text": "血压升高",
                "entity_type": "症状"
            },
            {
                "entity_end_pos": 14,
                "entity_start_pos": 12,
                "entity_text": "胸痛",
                "entity_type": "症状"
            },
            {
                "entity_end_pos": 12,
                "entity_start_pos": 10,
                "entity_text": "反复",
                "entity_type": "频率"
            },
            {
                "entity_end_pos": 9,
                "entity_start_pos": 6,
                "entity_text": "20年",
                "entity_type": "持续时长"
            },
            {
                "entity_end_pos": 17,
                "entity_start_pos": 14,
                "entity_text": "3年余",
                "entity_type": "持续时长"
            }
        ],
        "relation_list": [
            {
                "end_entity_end_pos": 9,
                "end_entity_start_pos": 6,
                "end_entity_text": "20年",
                "end_entity_type": "持续时长",
                "relation_type": "持续时间",
                "start_entity_end_pos": 6,
                "start_entity_start_pos": 2,
                "start_entity_text": "血压升高",
                "start_entity_type": "症状"
            },
            {
                "end_entity_end_pos": 17,
                "end_entity_start_pos": 14,
                "end_entity_text": "3年余",
                "end_entity_type": "持续时长",
                "relation_type": "持续时间",
                "start_entity_end_pos": 14,
                "start_entity_start_pos": 12,
                "start_entity_text": "胸痛",
                "start_entity_type": "症状"
            },
            {
                "end_entity_end_pos": 14,
                "end_entity_start_pos": 12,
                "end_entity_text": "胸痛",
                "end_entity_type": "症状",
                "relation_type": "频率修饰",
                "start_entity_end_pos": 12,
                "start_entity_start_pos": 10,
                "start_entity_text": "反复",
                "start_entity_type": "频率"
            }
        ],
        "text": "发现血压升高20年，反复胸痛3年余"
    }
    print(re_reader.__negative_sampling(info_dict))
