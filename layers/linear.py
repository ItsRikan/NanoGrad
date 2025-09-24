from engine import Matrics
import numpy as np

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

class BatchNorm1D:
    def __init__(self,fan_in,fan_out,changing_rate:float=0.1,eps=1e-3):
        assert 0<=changing_rate<=1 , "changing rate can't be less than 0 or more than 1"
        self.eps=eps
        self.cr=changing_rate
        self.beeta=Matrics(np.zeros((fan_out,)))
        self.gemma=Matrics(np.ones((fan_in,fan_out)))
        self.running_mean=None
        self.running_std=None
        self.training=True
    def __call__(self,x):
        if self.training:
            mean=x.mean(0,keepdims=True)
            var=x.var(0,keepdims=True)
            std=Matrics(np.sqrt(var.matrics + self.eps))
            if self.running_mean is None and self.running_std is None: 
                self.running_mean=mean.matrics
                self.running_std = std.matrics
        else:
            mean= Matrics(self.running_mean)
            std=Matrics(self.running_std)
        normalized = ( x - mean )/ std
        out = (self.gemma * normalized) + self.beeta
        if self.training:
            self.running_mean = (1-self.cr) * self.running_mean + self.cr * mean.matrics
            self.running_std = (1-self.cr) * self.running_std + self.cr * std.matrics
        return out
    def parameters(self):
        return [self.gemma,self.beeta]