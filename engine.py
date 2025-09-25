import numpy as np 
from util.util import fix_dim,set_grad
class Matrics:
    def __init__(self,matrics,child=tuple()):
        self.matrics=np.array(matrics,dtype=np.float32)
        self.grad=None
        self.shape=self.matrics.shape
        self._backward= lambda:None
        self.child=child
        self._backward = lambda: None
    def to_list(self):
        return self.matrics.tolist()
    def to_matrics(self,matrics):
        if isinstance(matrics,Matrics):
            return matrics
        elif isinstance(matrics,(np.ndarray,list)):
            return Matrics(matrics)
        elif isinstance(matrics,(int,float)):
            return Matrics([matrics])
        else:
            raise TypeError("must be list,numpy array or matrics")
    def __add__(self,other):
        other=self.to_matrics(other)
        out=Matrics(self.matrics + other.matrics,(self,other))
        def _backward():
            set_grad(self,out.grad)
            set_grad(other,out.grad)
        out._backward=_backward
        return out
    def __mul__(self,other):
        other=self.to_matrics(other)
        out=Matrics(self.matrics * other.matrics, (self,other))
        def _backward():
            set_grad(self,other.matrics * out.grad)
            set_grad(other,self.matrics * out.grad)

        out._backward=_backward
        return out
    def mean(self,dim:int=None,keepdims=False):
        out=Matrics(self.matrics.mean(dim,keepdims=keepdims),(self,))
        def _backward():
            set_grad(self,(1/np.prod(self.shape) * out.grad))
            
        out._backward = _backward
        return out
    def var(self,dim:int=None,keepdims=False):
        mean = self.mean(dim,keepdims)
        n = np.prod(self.shape) if dim is None else self.shape[dim]
        diff = self-mean
        out = (diff * diff).mean(dim,keepdims)
        return out
    def log(self):
        out=Matrics(np.log(self.matrics),(self,))
        def _backward():
            set_grad(self,(1/self.matrics) * out.grad )
        out._backward = _backward
        return out
    def __matmul__(self,other):
        other=self.to_matrics(other)
        out = Matrics((self.matrics @ other.matrics),(self,other))
        def _backward():
            set_grad(self,(out.grad @ other.matrics.T))
            set_grad(other,(self.matrics.T @ out.grad))
            
        out._backward=_backward
        return out
    def __pow__(self,power):
        assert isinstance(power,(int,float)),f"power can't be {type(power)} type"
        out = Matrics(np.power(self.matrics,power),(self,))
        def _backward():
            set_grad(self,power * np.power(self.matrics,power-1) * out.grad)
           
        out._backward=_backward
        return out
    def tanh(self):
        tanh_matrics= np.tanh(self.matrics)
        out=Matrics(tanh_matrics,(self,))
        def _backward():
            set_grad(self,(1-out.matrics**2) * out.grad)
        out._backward=_backward
        return out
    def sigmoid(self):
        neg_exp=np.exp(-1 * self.matrics)
        sig_matrics=1/(1+neg_exp)
        out=Matrics(sig_matrics,(self,))
        def _backward():
            set_grad(self,sig_matrics * (1-sig_matrics) * out.grad)
        out._backward=_backward
        return out
    def softmax(self,dim:int=-1):
        e_x=np.exp(self.matrics-np.max(self.matrics,axis=dim,keepdims=True))
        soft_matrics=e_x / e_x.sum(axis=dim,keepdims=True)
        out=Matrics(soft_matrics,(self,))
        def _backward():
            s=out.matrics
            upstream_grad=out.grad
            s_spread = s[...,np.newaxis]
            grad = s * (upstream_grad - (upstream_grad*s).sum(axis=dim,keepdims=True))
            set_grad(self,grad)
        out._backward=_backward
        return out
               
    def ReLU(self,thershold:float=0):
        assert 0<=thershold<=1 , "thershold can't be greater than 1 or less than 0"
        relu_matrics=np.maximum(self.matrics,self.matrics*thershold)
        out=Matrics(relu_matrics,(self,))
        def _backward():
            set_grad(self,np.where(relu_matrics>0,1,thershold) * out.grad)    
        out._backward=_backward
        return out   
    def exp(self):
        out = Matrics(np.exp(self.matrics),(self,))
        def _backward():
            set_grad(self,out.matrics * out.grad)
        out._backward=_backward
        return out
    
    def item(self):
        assert np.prod(self.shape)==1 , "can't convert a matrics to single item"
        return self.matrics.tolist()
    def backward(self):
        topology=[]
        visited=set()
        def create_topo(parent):
            if parent not in visited:
                for ch in parent.child:
                    create_topo(ch)
                topology.append(parent)
                visited.add(parent)
        create_topo(self)
        self.grad=np.ones_like(self.matrics)
        for node in reversed(topology): 
            node._backward()
                

    def __neg__(self):
        return -1 * self
    def __rmul__(self,other):
        return self * other
    def __sub__(self,other):
        return self + (-other)
    def __rsub__(self,otehr):
        return other +(-self)
    def __radd__(self,other):
        return self + other
    def __truediv__(self,other):
        return self * other**-1
    def __rtruediv__(self,other):
        return other * self**-1
    def __rmatmul__(self,other):
        return self @ other
    def __repr__(self):
        return str(self.to_list())