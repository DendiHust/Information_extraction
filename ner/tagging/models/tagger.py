#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ner 
@File    ：tagger.py
@Author  ：hbx
@Date    ：2021/12/6 16:33 
'''
import torch
from overrides import overrides
from torch.nn import Linear
from typing import Dict, List, Optional
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.initializers import InitializerApplicator
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
import torch.nn.functional as F


@Model.register('tagger')
class Tagger(Model):
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

        if dropout is not None:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._feedforward = feedforward

        if self._feedforward:
            output_dim = self._feedforward.get_output_dim()
        elif self._encoder:
            output_dim = self._encoder.get_output_dim()
        else:
            output_dim = self._embedder.get_output_dim()

        self._tag_projection_layer = TimeDistributed(Linear(in_features=output_dim, out_features=self._num_tags))

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

        mask = get_text_field_mask(tokens)
        tmp_encoded = self._embedder(tokens)
        if self._encoder is not None:
            tmp_encoded = self._encoder(tmp_encoded, mask)

        if self._feedforward is not None:
            tmp_encoded = self._feedforward(tmp_encoded)

        if self._dropout is not None:
            tmp_encoded = self._dropout(tmp_encoded)

        logits = self._tag_projection_layer(tmp_encoded)

        if self._use_crf:

            best_paths = self._crf.viterbi_tags(logits, mask)
            predicted_tags = [path for path, score in best_paths]

            # Just get the top tags and ignore the scores.
            # predicted_tags = cast(List[List[int]], [x[0][0] for x in viterbi_tags])

            broadcasted = self._broadcast_tags(predicted_tags, logits)

            output = {
                'logits': logits,
                'tags': predicted_tags
            }

            if labels is not None:
                log_likelihood = self._crf(logits, labels, mask)

                output['loss'] = -log_likelihood
                self._f1(broadcasted, labels, mask)

        else:
            predicted_tags = F.softmax(logits, dim=-1)
            output = {"logits": logits, "tags": predicted_tags}
            if labels is not None:
                loss = sequence_cross_entropy_with_logits(logits, labels, mask)
                output['loss'] = loss
                self._f1(logits, labels, mask)

        return output