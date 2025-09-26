import numpy as np
from engine import Matrics
from util.util_for_file import save_obj,to_json,from_json
class Sequential:
    def __init__(self,layers:list=[]):
        self.layers=layers
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    def save(self,file_path:str='artifacts/model.pkl'):
        save_obj(self,file_path)
    def save_params(self,file_path:str='artifacts/params.json'):
        params=[p.matrics.tolist() for layer in self.layers for p in layer.parameters()]
        to_json(params,file_path)
    def from_pretrained(self,file_path:str='artifacts/params.json'):
        params = from_json(file_path)
        set_params = [p for layer in self.layers for p in layer.parameters()]
        assert isinstance(params,list) , "only list is acceptable"
        if len(set_params) != len(params):
            raise IndexError("loaded params index mismatch")
        for i in range(len(params)):
            matrics=np.array(params[i])
            if matrics.shape == set_params[i].shape:
                set_params[i].matrics =  matrics
            else:
                raise IndexError("loaded params dimension mismatch")
    def parameters(self):
        params=[p for layer in self.layers for p in layer.parameters()]
        return params
