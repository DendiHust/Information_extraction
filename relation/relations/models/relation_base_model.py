'''
 # @ Author: xuan
 # @ Create Time: 2021-08-30 15:45:19
 # @ Modified by: xuan
 # @ Modified time: 2021-08-30 16:48:14
 # @ Description:
 '''

from typing import Dict, Optional, List
from allennlp.modules.token_embedders.embedding import Embedding
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules import FeedForward
from allennlp.nn.util import get_text_field_mask

from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from ..nn.focal_loss import FocalLoss


@Model.register('relation_base_model')
class RelationBaseModel(Model):

    def __init__(self, vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 entity_embedder: Embedding,
                 encoder: Seq2SeqEncoder,
                 attention: Seq2SeqEncoder,
                 loss_func: str = None,
                 **kwargs
                 ) -> None:
        super().__init__(vocab, **kwargs)

        self._embedder = embedder
        self._entity_embedder = entity_embedder
        self._encoder = encoder
        self._attention = attention
        self._feed_fward = torch.nn.Linear(
            in_features=attention.get_output_dim(), out_features=vocab.get_vocab_size('labels'))
        self._all_f1 = FBetaMeasure(average='micro')
        self._f1 = FBetaMeasure()
        # self._metrics = {
        #     'all_f1': FBetaMeasure(average='micro'),
        #     'class_f1': FBetaMeasure()
        # }
        if loss_func == 'focal_loss':
            self._loss = FocalLoss()
        else:
            self._loss = torch.nn.CrossEntropyLoss()

    @overrides
    def get_metrics(self, reset: bool) -> Dict[str, float]:
        # {name: metric.get_metric() for name, metric in self._f1.items()}
        # return self._all_f1.get_metric(reset)
        f1_dict = self._f1.get_metric(reset)
        output = {}
        # output['accuracy'] = self._accuracy.get_metric(reset=reset)
        counter = 0
        for precision, recall, fscore in zip(f1_dict['precision'], f1_dict['recall'], f1_dict['fscore']):
            output[self.vocab.get_index_to_token_vocabulary(
                'labels')[counter] + '_precision'] = precision
            output[self.vocab.get_index_to_token_vocabulary(
                'labels')[counter] + '_recall'] = recall
            output[self.vocab.get_index_to_token_vocabulary(
                'labels')[counter] + '_fscore'] = fscore
            counter += 1
        output.update(self._all_f1.get_metric(reset))
        return output

    def forward(self, tokens: Dict[str, torch.Tensor],
                seq_entity_tags: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(tokens)
        token_embedded = self._embedder(tokens)
        entity_embedded = self._entity_embedder(seq_entity_tags)

        inputs = torch.cat([token_embedded, entity_embedded], dim=-1)

        encoded = self._encoder(inputs, mask)

        attened = self._attention(encoded, mask)

        attened = torch.mean(attened, dim=1)

        logits = self._feed_fward(attened)

        probs = F.softmax(logits, dim=-1)

        output = {
            'logits': logits,
            'probs': probs
        }

        if labels is not None:
            loss = self._loss(logits, labels)
            output['loss'] = loss
            self._all_f1(logits, labels)
            self._f1(logits, labels)

        return output


    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict["probs"]
        predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary('labels').get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["predict_label"] = classes
        return output_dict

