#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：span_util.py
@Author  ：hbx
@Date    ：2022/1/6 10:42 
'''
from typing import Tuple, List, Set


TypedStringSpan = Tuple[str, Tuple[int, int]]


def mrc_decode(
        start_predictions: List[int],
        end_predictions: List[int],
        span_type: str
    ) -> List[TypedStringSpan]:
    spans: Set[TypedStringSpan] = set()

    for _index, s_tag in enumerate(start_predictions):
        if s_tag == 0:
            continue
        for _offset, e_tag in enumerate(end_predictions[_index:]):
            if e_tag == 1:
                spans.add((span_type, (_index, _index + _offset)))
                break
    return list(spans)