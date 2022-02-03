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
from allennlp.nn.initializers import InitializerApplicator

from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from ..nn.focal_loss import FocalLoss


@Model.register('relation_base_model')
class RelationBaseModel(Model):

    def __init__(self, vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 dropout: Optional[float] = None,
                 loss_func: str = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs
                 ) -> None:
        super().__init__(vocab, **kwargs)

        self._embedder = embedder
        if encoder:
            self._encoder = encoder
        else:
            self._encoder = None
        if dropout:
            self._dropout = torch.nn.Dropout()
        else:
            self._dropout = None
        output_dim = self._embedder.get_output_dim()
        if self._encoder:
            output_dim = self._encoder.get_output_dim()

        self._feedforward = torch.nn.Linear(
            in_features=output_dim * 2, out_features=vocab.get_vocab_size('labels'))
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

        initializer(self)

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
                subj_start_index: torch.Tensor,
                obj_start_index: torch.Tensor,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(tokens)
        sequence_output = self._embedder(tokens)
        if self._encoder:
            sequence_output = self._encoder(sequence_output, mask)
        if self._dropout:
            sequence_output = self._dropout(sequence_output)

        subj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subj_start_index)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_start_index)])
        rep = torch.cat([subj_output, obj_output], dim=1)

        logits = self._feedforward(rep)

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
