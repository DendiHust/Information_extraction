'''
 # @ Author: xuan
 # @ Create Time: 2021-09-07 16:15:25
 # @ Modified by: xuan
 # @ Modified time: 2021-09-07 16:15:33
 # @ Description:
 '''
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor
from allennlp.data import Instance


@Predictor.register('crf_predictor')
class CRFPredictor(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict['words'])

    def predict(self, words: str) -> JsonDict:
        return self.predict_json({'words': words})
