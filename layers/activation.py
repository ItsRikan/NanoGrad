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
class LeakyReLU:
    def __call__(self,x):
        x=to_matrics(x)
        out=x.ReLU(thershold=0.01)
        return out
    def parameters(self):
        return []
class PReLU:
    def __call__(self,x,alpha:float=0.01):
        x=to_matrics(x)
        out=x.ReLU(thershold=alpha)
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

class Sigmoid:
    def __call__(self,x):
        x=to_matrics(x)
        out=x.sigmoid()
        return out
    def parameters(self):
        return []
class Softmax:
    def __init__(self,dim:int=-1):
        self.dim=dim
    def __call__(self,x):
        x=to_matrics(x)
        out=x.softmax(self.dim)
        return out
    def parameters(self):
        return []