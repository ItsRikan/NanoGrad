from engine import Matrics
import numpy as np
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
class Linear:
    def __init__(self,fan_in,fan_out,bias=True):
        self.bias=bias
        limit=np.sqrt(6/(fan_in + fan_out))
        self.w = Matrics(np.random.uniform(-limit,limit,(fan_in,fan_out)))
        if self.bias:
            self.b = Matrics(np.zeros((fan_out,)))
    def __call__(self,X):
        out= X @ self.w
        
        if self.bias:
            out += self.b
        return out
    def parameters(self):
        return [self.w,self.b] if self.bias else [self.w]