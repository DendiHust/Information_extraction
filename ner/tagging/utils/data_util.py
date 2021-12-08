
from typing import Dict, Text


def convert_2_crf_example(data_item: Dict):
    '''
    description: 将元数据转为序列标注的格式
    param {*}
    return {*}
    '''
    text = data_item['text']
    labels = ['O'] * len(text)
    print(data_item)
    for entity_item in data_item['labels']:
        for t_index in range(entity_item[2], entity_item[3]):
            if t_index == entity_item[2]:
                labels[t_index] = 'B-' + entity_item[1]
            else:
                labels[t_index] = 'I-' + entity_item[1]
    assert len(text) == len(labels)
    return {'text': text, 'labels': labels}


if __name__ == '__main__':
    from pathlib import Path
    print(Path.cwd())
    with open(Path.cwd().joinpath('ner', 'data', 'raw_data', 'dev.json'), mode='r', encoding='utf8') as f:
        import json
        data_ = json.load(f)
        tmp = convert_2_crf_example(data_[0])
        for t, l in zip(tmp['text'], tmp['labels']):
            print(f'{t}\t{l}')
        
    
