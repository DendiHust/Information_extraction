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
from torch.nn import Linear
from typing import Dict, List, Optional
from allennlp.data.vocabulary import Vocabulary
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn import util
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
import torch.nn.functional as F


@Model.register('bert_crf')
class BertCrf(Model):
    def __init__(self, vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 constraint_crf_decoding: bool = True,
                 feedforward: Optional[FeedForward] = None,
                 dropout: Optional[float] = None,
                 label_namespace: str = 'labels',
                 use_pretrained_model: Optional[bool] = None,
                 label_encoding: str = 'BIO',
                 use_crf: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs):
        super().__init__(vocab, **kwargs)

        self._num_tags = vocab.get_vocab_size('labels')
        self._embedder = embedder

        self._encoder = encoder
        self._constrain_crf_decoding = constraint_crf_decoding
        self._use_pretrained_model = use_pretrained_model
        self._use_crf = use_crf
        self._label_encoding = label_encoding
        self._label_namespace = label_namespace

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._feedforward = feedforward

        if self._feedforward:
            output_dim = self._feedforward.get_output_dim()
        else:
            output_dim = self._embedder.get_output_dim()

        self._classifier = TimeDistributed(Linear(in_features=output_dim, out_features=self._num_tags))

        if self._use_crf:
            if constraint_crf_decoding:
                labels = vocab.get_index_to_token_vocabulary(self._label_namespace)
                constraints = allowed_transitions(self._label_encoding, labels)
            else:
                constraints = None
            self._crf = ConditionalRandomField(vocab.get_vocab_size('labels'), constraints)

        self._f1 = SpanBasedF1Measure(vocab, self._label_namespace)

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
                self.vocab.get_token_from_index(tag, namespace=self._label_namespace) for tag in tags
            ]

        output_dict["tags"] = [decode_tags(t) for t in output_dict["tags"]]
        return output_dict

    def forward(self, tokens: Dict[str, Dict[str, torch.Tensor]], labels = None) -> Dict[str, torch.Tensor]:
        # mask = tokens['tokens']['mask']
        # bert_embeddings, _ = self._bert_model(
        #     input_ids=tokens['tokens']['token_ids'],
        #     token_type_ids=tokens['tokens']['type_ids'],
        #     attention_mask=mask,
        #     return_dict=False,
        # )
        mask = get_text_field_mask(tokens)
        tmp_encoded = self._embedder(tokens)
        if self._encoder:
            tmp_encoded = self._encoder(tmp_encoded, mask)

        if self._feedforward:
            tmp_encoded = self._feedforward(tmp_encoded)

        if self._dropout:
            tmp_encoded = self._feedforward(tmp_encoded)
        classified = self._classifier(tmp_encoded)
        if self._use_crf:

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


        else:
            class_probabilities = F.softmax(classified, dim=-1)
            output = {"logits": classified, "class_probabilities": class_probabilities}
            if labels is not None:
                loss = sequence_cross_entropy_with_logits(classified, labels, mask)
                output['loss'] = loss
                self._f1(classified, labels, mask)

        return output