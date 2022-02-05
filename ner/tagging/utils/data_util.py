import json
from typing import Dict, List
from collections import defaultdict
import numpy as np


def convert_2_tagger_example(data_item: Dict) -> Dict:
    '''
    description: 将元数据转为序列标注的格式
    param {*}
    return {*}
    '''
    text = data_item['text']
    labels = ['O'] * len(text)
    # print(data_item)
    for entity_item in data_item['labels']:
        for t_index in range(entity_item[2], entity_item[3]):
            if t_index == entity_item[2]:
                labels[t_index] = 'B-' + entity_item[1]
            else:
                labels[t_index] = 'I-' + entity_item[1]
    assert len(text) == len(labels)
    return {'text': text, 'labels': labels}


def convert_2_mrc_exapmple(data_item: Dict, question_dict: Dict = None) -> Dict:
    """ 将元数据转为阅读理解的格式 """
    if question_dict is None:
        with open('./data/mid_data/mrc_ent2id.json', mode='r', encoding='utf8') as f:
            question_dict = json.load(f)

    text = data_item['text']
    label_dict = defaultdict(list)
    for _entity in data_item['labels']:
        # (entity_start_pos, entity_end_pos, entity_text)
        label_dict[_entity[1]].append((_entity[2], _entity[3], _entity[4]))
    labels = []
    for _type in question_dict.keys():
        start_labels = [0] * len(text)
        end_labels = [0] * len(text)

        tmp_label = {}
        tmp_label['question'] = question_dict[_type]
        tmp_label['question_type'] = _type

        for _label in label_dict[_type]:
            start_labels[_label[0]] = 1
            end_labels[_label[1] - 1] = 1

        tmp_label['start_labels'] = start_labels
        tmp_label['end_labels'] = end_labels

        labels.append(tmp_label)

    return {'text': text, 'labels': labels}


def convert_2_global_pointer_example(data_item: Dict, entity_id_dict: Dict = None) -> Dict:
    if entity_id_dict is None:
        with open('./data/mid_data/global_ent2id.json', mode='r', encoding='utf8') as f:
            entity_id_dict = json.load(f)
    entity_type_number = len(entity_id_dict)
    text = data_item['text']
    seq_length = len(text)
    labels = np.zeros((entity_type_number, seq_length, seq_length))
    for entity_item in data_item['labels']:
        labels[entity_id_dict[entity_item[1]], entity_item[2], entity_item[3] - 1] = 1
    return {'text': text, 'labels': labels}


if __name__ == '__main__':
    from pathlib import Path

    print(Path.cwd())
    # with open(Path.cwd().joinpath('ner', 'data', 'raw_data', 'dev.json'), mode='r', encoding='utf8') as f:
    #     import json
    #     data_ = json.load(f)
    #     tmp = convert_2_tagger_example(data_[0])
    #     for t, l in zip(tmp['text'], tmp['labels']):
    #         print(f'{t}\t{l}')

    with open('../../data/raw_data/tmp.json', mode='r', encoding='utf8') as f:
        data = json.load(f)

    # with open('../../data/mid_data/mrc_ent2id.json', mode='r', encoding='utf8') as f:
    #     quesion_dict = json.load(f)
    #
    # tmp_data = convert_2_mrc_exapmple(data[0], quesion_dict)
    # # print(tmp_data)
    # from tagging.utils.span_util import mrc_decode
    #
    # decode = mrc_decode(tmp_data['labels'][2]['start_labels'], tmp_data['labels'][2]['end_labels'],
    #                     tmp_data['labels'][2]['question_type'])
    # print(decode)

    with open('../../data/mid_data/global_ent2id.json', mode='r', encoding='utf8') as f:
        entity_id_dict = json.load(f)
    tmp_data = convert_2_global_pointer_example(data[0], entity_id_dict)
    print(tmp_data)
