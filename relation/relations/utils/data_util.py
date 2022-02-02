#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：relation 
@File    ：data_util.py
@Author  ：hbx
@Date    ：2022/2/2 17:34 
'''
from typing import List, Dict, Any


def get_text_with_entity_type_tag(
        text: str,
        start_entity_start_index: int,
        start_entity_end_index: int,
        start_entity_type: str,
        end_entity_start_index: int,
        end_entity_end_index: int,
        end_entity_type:str
) -> Dict[str, Any]:

    token_list = []
    subj_start_index, subj_end_index, obj_start_index, obj_end_index = 0, 0, 0, 0
    for i, token in enumerate(text):
        if i == start_entity_start_index:
            subj_start_index = len(token_list)
            token_list.append(f"<S: {start_entity_type.upper()}>")

        if i == end_entity_start_index:
            obj_start_index = len(token_list)
            token_list.append(f"<O: {end_entity_type.upper()}>")

        token_list.append(token)

        if i == start_entity_end_index - 1:
            subj_end_index = len(token_list)
            token_list.append(f"</S: {start_entity_type.upper()}>")

        if i == end_entity_end_index - 1:
            obj_end_index = len(token_list)
            token_list.append(f"</O: {end_entity_type.upper()}>")

    # if end_entity_start_index > start_entity_end_index:
    #     end_entity_start_index += 2
    #     end_entity_end_index += 2
    #     token_list.insert(start_entity_start_index, f"<S: {start_entity_type.upper()}>")
    #     token_list.insert(start_entity_end_index + 1, f"</S: {start_entity_type.upper()}>")
    #     token_list.insert(end_entity_start_index, f"<O: {end_entity_type.upper()}>")
    #     token_list.insert(end_entity_end_index + 1, f"</O: {end_entity_type.upper()}>")
    # elif end_entity_end_index < start_entity_start_index:
    #     token_list.insert(end_entity_start_index, f"<O: {end_entity_type.upper()}>")
    #     token_list.insert(end_entity_end_index + 1, f"</O: {end_entity_type.upper()}>")
    #     start_entity_start_index += 2
    #     start_entity_end_index += 2
    #     token_list.insert(start_entity_start_index, f"<S: {start_entity_type.upper()}>")
    #     token_list.insert(start_entity_end_index + 1, f"</S: {start_entity_type.upper()}>")

    return {'tokens': token_list, 'subj_start_index': subj_start_index, 'obj_start_index': obj_start_index}

if __name__ == '__main__':
    text = "反复胸闷、心悸20余年，再发1月余"
    tmp = get_text_with_entity_type_tag(
        text=text,
        end_entity_start_index=0,
        end_entity_end_index=2,
        start_entity_type='频率',
        start_entity_start_index=5,
        start_entity_end_index=7,
        end_entity_type='症状'
    )
    print(tmp)
