'''
 # @ Author: xuan
 # @ Create Time: 2021-08-23 18:31:33
 # @ Modified by: xuan
 # @ Modified time: 2021-08-23 18:32:03
 # @ Description:
 '''

from os import path
from typing import Dict, Optional, List, cast
from numpy import broadcast
from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits


from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure


@Model.register('crf_tragger')
class CrfTragger(Model):

    def __init__(self, vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs) -> None:
        super().__init__(vocab, **kwargs)
        self._embedder = embedder
        self._encoder = encoder
        self._classifier = torch.nn.Linear(
            in_features=encoder.get_output_dim(), out_features=vocab.get_vocab_size('labels'))
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

        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)
        classified = self._classifier(encoded)

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
