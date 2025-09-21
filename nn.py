from engine import Matrics
import numpy as np
class Linear:
    def __init__(self,fan_in,fan_out,bias=True):
        self.bias=bias
        self.w = Matrics(np.random.uniform(-1,1,(fan_in,fan_out)))
        if self.bias:
            self.b = Matrics(np.zeros((fan_out,)))
    def __call__(self,X):
        out= X @ self.w
        if self.bias:
            out += self.b
        return out
    def parameters(self):
        return [self.w.matrics,self.b.matrics] if bias else [self.w.matrics]