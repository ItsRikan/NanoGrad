import numpy as np
from engine import Matrics


class SGD:
    def __init__(self,params:list=[],learnning_rate:float=0.01):
        self.params=params
        self.lr=learnning_rate
    def zero_grad(self,set_to_none:bool=True):
        for p in self.params:
            p.grad=None
    def step(self):
        for p in self.params:
            if p.grad is not None:
                clipped_grad = np.clip(p.grad,-0.1,0.1)
                p.matrics -= self.lr * p.grad