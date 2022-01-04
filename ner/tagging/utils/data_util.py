
from typing import Dict, List
from collections import defaultdict


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

def convert_2_mrc_exapmple(data_item: Dict, question_dict: Dict) -> Dict:
    """ 将元数据转为阅读理解的格式 """
    text = data_item['text']
    label_dict = defaultdict(list)
    for _entity in data_item['labels']:
        # (entity_start_pos, entity_end_pos, entity_text)
        label_dict[_entity[1]].append((_entity[1], _entity[2], _entity[3]))
    for _type in question_dict.keys():
        start_labels = [0] * len(text)
        end_labels = [0] * len(text)

        pass


if __name__ == '__main__':
    from pathlib import Path
    print(Path.cwd())
    with open(Path.cwd().joinpath('ner', 'data', 'raw_data', 'dev.json'), mode='r', encoding='utf8') as f:
        import json
        data_ = json.load(f)
        tmp = convert_2_tagger_example(data_[0])
        for t, l in zip(tmp['text'], tmp['labels']):
            print(f'{t}\t{l}')
        
    
