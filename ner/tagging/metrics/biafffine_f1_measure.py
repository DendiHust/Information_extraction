#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：biafffine_f1_measure.py
@Author  ：hbx
@Date    ：2022/2/11 17:53 
'''
import torch
from allennlp.training.metrics import Metric
from overrides import overrides
from collections import defaultdict
from typing import Dict, Optional, Any, Set


@Metric.register('biaffine_f1')
class BiaffineMeasure(Metric):
    def __init__(self, entity_id_dict: Dict):
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)
        self._id_entity_dict = {v: k for k, v in entity_id_dict.items()}

    @overrides
    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        # (batch_size, entity_type_size, seq_length, seq_length)
        batch_size = predictions.shape[0]
        for i in range(batch_size):
            pre_item = predictions[i]
            gold_labels_item = gold_labels[i]
            # 分别统计每种实体类型的指标
            for entity_id, entity_type in self._id_entity_dict.items():
                pre_entity_item = torch.eq(pre_item, entity_id)
                gold_entity_labels_item = torch.eq(gold_labels_item, entity_id)

                self._true_positives[self._id_entity_dict[entity_id]] += torch.sum(
                    pre_entity_item * gold_entity_labels_item).cpu().numpy()
                self._false_positives[self._id_entity_dict[entity_id]] += torch.sum(
                    pre_entity_item * (1 - gold_entity_labels_item.float())).cpu().numpy()
                self._false_negatives[self._id_entity_dict[entity_id]] += torch.sum(
                    (1 - pre_entity_item.float()) * gold_entity_labels_item.float()).cpu().numpy()

    @overrides
    def get_metric(self, reset: bool) -> Dict[str, Any]:
        all_metrics = {}
        for id, entity_type in self._id_entity_dict.items():
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[entity_type], self._false_positives[entity_type],
                self._false_negatives[entity_type]
            )
            precision_key = "precision" + "-" + entity_type
            recall_key = "recall" + "-" + entity_type
            f1_key = "f1-measure" + "-" + entity_type
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure
        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure

        if reset:
            self.reset()
        return all_metrics

    @overrides
    def reset(self) -> None:
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = true_positives / (true_positives + false_positives + 1e-13)
        recall = true_positives / (true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * (precision * recall) / (precision + recall + 1e-13)
        return precision, recall, f1_measure
