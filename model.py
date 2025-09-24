import numpy as np
from engine import Matrics

class Sequential:
    def __init__(self,layers:list=[]):
        self.layers=layers
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    def parameters(self):
        params=[p for layer in self.layers for p in layer.parameters()]
        return params
