import numpy as np
from util.util import set_grad,fix_dim
from util.to_matrics import to_matrics
class ReLU:
    def __call__(self,x):
        x=to_matrics(x)
        out=x.ReLU()
        return out
    def parameters(self):
        return []

class Tanh:
    def __call__(self,x):
        x=to_matrics(x)
        out=x.tanh()
        return out
    def parameters(self):
        return []