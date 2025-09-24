import numpy as np
from engine import Matrics


class SGD:
    def __init__(self,params:list=[],learnning_rate:float=0.01,momentum_decay:float=0,initial_momentum:int=0):
        self.params=params
        self.lr=learnning_rate
        self.momentum_decay=momentum_decay
        self.velocity = np.array([np.ones_like(p.matrics) for p in self.params]) * initial_momentum
    def zero_grad(self,set_to_none:bool=False):
        if set_to_none:
            for p in self.params:
                p.grad=None
        else:
            for p in self.params:
                p.grad = np.zeros(p.shape)
    def step(self):
        for i,p in enumerate(self.params):
            if p.grad is not None:
                self.velocity[i] = (self.velocity[i] * self.momentum_decay) + (self.lr * p.grad)
                p.matrics -= self.lr * self.velocity[i]