import numpy as np


class Adam:
    def __init__(self,params:list=[],learnning_rate:float=0.01,eps:float=1e-3,momentum_decay:float=0.9,initial_momentum:float=0,velocity_decay:float=0.9,initial_velocity:float=0):
        self.params=params
        self.lr=learnning_rate
        self.momentum = [np.ones_like(p.matrics)* initial_momentum for p in self.params] 
        self.velocity = [np.ones_like(p.matrics)* initial_velocity for p in self.params] 
        assert 0<=momentum_decay<=1 and 0<=velocity_decay<=1, "decays can't be less than 0 and greater than 1"
        self.momentum_decay=momentum_decay
        self.velocity_decay=velocity_decay
        self.eps=eps
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
                self.velocity[i] = self.velocity[i] * self.velocity_decay + (1-self.velocity_decay) * np.power(p.grad,2)
                self.momentum[i] = self.momentum[i] * self.momentum_decay + (1-self.momentum_decay) * p.grad
                p.matrics -= self.lr * self.momentum / np.sqrt(self.velocity + self.eps)
