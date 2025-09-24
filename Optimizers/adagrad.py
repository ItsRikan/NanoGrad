import numpy as np

class AdaGrad:
    def __init__(self,params:list=[],learnning_rate:float=0.01,initial_velocity:float=0,eps:float=1e-3,velocity_decay:int=0):
        self.params = params
        self.eps=eps
        self.velocity = np.array([np.ones_like(p.matrics) for p in self.params]) * initial_velocity
        self.lr=learnning_rate
        assert 0<=velocity_decay<=1 ,"velocity decay can't be less than 0 or greater than 1"
        self.bitta=velocity_decay

    def zero_grad(self,set_to_none:bool=False):
        if set_to_none:
            for p in self.params:
                p.grad=None
        else:
            for p in self.params:
                p.grad = np.zeros(p.shape)
    def step(self):
        for i,p in enumerate(self.params):
            self.velocity[i] = self.velocity[i]*self.bitta + (1-self.bitta) * np.power(p.grad,2)
            p.matrics -= self.lr * p.grad / np.sqrt(self.velocity[i] + self.eps)