'''
 # @ Author: xuan
 # @ Create Time: 2021-09-09 10:02:46
 # @ Modified by: xuan
 # @ Modified time: 2021-09-09 10:02:49
 # @ Description:
 '''

from typing import Any, Dict
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor
from allennlp.data.instance import Instance

@Predictor.register('relation_base_predictor')
class RelationBasePredictor(Predictor):
    
    def predict(self, entity_info:JsonDict) -> JsonDict:
        for item in entity_info['ner_list']:
            tmp_input = {}
            
        # self.predict_json(inputs)


        pass

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        

        return super()._json_to_instance(json_dict)




