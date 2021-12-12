#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：bert_crf.py
@Author  ：hbx
@Date    ：2021/12/6 16:33 
'''
import torch
from overrides import overrides
from typing import Dict, List
from allennlp.data.vocabulary import Vocabulary
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn import util
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure


@Model.register('bert_crf')
class BertCrf(Model):
    def __init__(self, vocab: Vocabulary,
                 bert_path: str,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs):
        super().__init__(vocab, **kwargs)

        self._bert_config = BertConfig.from_pretrained(bert_path)
        self._bert_config.num_labels = vocab.get_vocab_size('labels')
        self._bert_model = BertModel(self._bert_config)

        self._classifier = torch.nn.Linear(
            in_features=self._bert_model.config.hidden_size, out_features=vocab.get_vocab_size('labels'))
        self._crf = ConditionalRandomField(vocab.get_vocab_size('labels'))
        self._f1 = SpanBasedF1Measure(vocab, 'labels')

        initializer(self)

    def _broadcast_tags(self,
                        viterbi_tags: List[List[int]],
                        logits: torch.Tensor) -> torch.Tensor:
        output = logits * 0.
        for i, sequence in enumerate(viterbi_tags):
            for j, tag in enumerate(sequence):
                output[i, j, tag] = 1.
        return output

    @overrides
    def get_metrics(self, reset: bool) -> Dict[str, float]:
        return self._f1.get_metric(reset)

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        def decode_tags(tags):
            return [
                self.vocab.get_token_from_index(tag, namespace='labels') for tag in tags
            ]

        output_dict["tags"] = [decode_tags(t) for t in output_dict["tags"]]
        return output_dict

    def forward(self, tokens: Dict[str, torch.Tensor], labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        bert_embeddings, _ = self._bert_model(
            input_ids=tokens['tokens']['token_ids'],
            token_type_ids=tokens['tokens']['type_ids'],
            attention_mask=mask,
            return_dict=False,
        )
        # encoded = self._encoder(embedded, mask)
        classified = self._classifier(bert_embeddings)

        viterbi_tags = self._crf.viterbi_tags(classified, mask)
        viterbi_tags = [path for path, score in viterbi_tags]

        # Just get the top tags and ignore the scores.
        # predicted_tags = cast(List[List[int]], [x[0][0] for x in viterbi_tags])

        broadcasted = self._broadcast_tags(viterbi_tags, classified)

        output = {
            'logits': classified,
            'tags': viterbi_tags
        }

        if labels is not None:
            log_likelihood = self._crf(classified, labels, mask)

            output['loss'] = -log_likelihood
            self._f1(broadcasted, labels, mask)

        return output
