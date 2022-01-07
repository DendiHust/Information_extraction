'''
@Project ：ner 
@File    ：mrc_f1_measure.py
@Author  ：hbx
@Date    ：2022/1/5 16:40 
'''
import torch
from allennlp.training.metrics import Metric
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from overrides import overrides
from typing import Dict, Optional, Any, Set
from collections import defaultdict
from tagging.utils.span_util import mrc_decode

@Metric.register('mrc_f1')
class MRCF1Measure(Metric):

    def __init__(self):
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    @overrides
    def __call__(
            self,
            start_predictions: torch.Tensor,
            end_predictions: torch.Tensor,
            gold_start_labels: torch.Tensor,
            gold_end_labels: torch.Tensor,
            metadata: dict,
            mask: Optional[torch.BoolTensor] = None
        ):
        start_predictions, end_predictions, gold_start_labels, gold_start_labels, mask = self.detach_tensors(
            start_predictions, end_predictions, gold_start_labels, gold_start_labels, mask
        )
        start_predictions_tmp = start_predictions.cpu().numpy().argmax(-1)
        end_predictions_tmp = end_predictions.cpu().numpy().argmax(-1)

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        batch_size = start_predictions.shape[0]
        for i in range(batch_size):
            length = sequence_lengths[i]
            span_type = metadata[i]['question_type']
            start_index = len(metadata[i]['question']) + 2

            if length == 0:
                continue

            sequence_start_prediction = start_predictions_tmp[i, start_index: length].tolist()
            sequence_end_prediction = end_predictions_tmp[i, start_index: length].tolist()


            sequence_start_gold_label = gold_start_labels[i, start_index: length].tolist()
            sequence_end_gold_label = gold_end_labels[i, start_index: length].tolist()

            predicted_spans = mrc_decode(sequence_start_prediction,sequence_end_prediction,span_type)
            gold_spans = mrc_decode(sequence_start_gold_label,sequence_end_gold_label,span_type)

            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1

            for span in gold_spans:
                self._false_negatives[span[0]] += 1



    @overrides
    def get_metric(self, reset: bool) -> Dict[str, Any]:
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
            )
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
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


