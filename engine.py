import numpy as np 

class Metrics:
    def __init__(self,metrics,child=tuple()):
        self.metrics=np.array(metrics,dtype=np.float32)
        #assert self.metrics.ndim==2,'only two dimension metrics is acceptable'
        self.grad=np.zeros_like(self.metrics,dtype=np.float32)
        self._backward= lambda:None
        self.child=child
    def __to_metrics__(self,metrics):
        if isinstance(metrics,Metrics):
            return metrics
        elif isinstance(metrics,(np.ndarray,list)):
            return Metrics(metrics)
        
        else:
            raise TypeError("must be list,numpy array or metrics")

    def to_list(self):
        return self.metrics.to_list()

    @staticmethod
    def __fix_dim__(match:np.ndarray,out:np.ndarray)->np.ndarray:
        match_shape=match.shape
        out_shape=out.shape
        if match_shape==out_shape:
            return out
        elif sum(out_shape)>sum(match_shape):
            if len(match_shape)==1:
                if match_shape[0]==out_shape[1]:
                    return out.sum(0)
                elif match_shape[0]==1:
                    return out.sum().reshape(1,)
            elif match_shape[0]==1 and match_shape[1]==1:
                return out.sum().reshape(1,1)
            elif match_shape[0]==out_shape[0] and match_shape[1]==1:
                return out.sum(1).reshape(match_shape[0],1)
            elif match_shape[1]==out_shape[1] and match_shape[0]==1:
                return out.sum(0).reshape(1,match_shape[1])
        elif sum(out_shape)<sum(match_shape):
            if len(out_shape)==1:
                if out_shape[0]==match_shape[1]:
                    return np.tile(out,(match_shape[0],1))
                elif out_shape[0]==1:
                    return np.tile(out,match_shape)
            elif match_shape[0]==1 and match_shape[1]==1:
                return np.tile(out,match_shape)
            elif out_shape[1]==match_shape[1] and out_shape[0]==1:
                return np.tile(out,(match_shape[0],1))
            elif out_shape[0]==match_shape[0] and out_shape[1]==1:
                return np.tile(out,(1,match_shape[1]))


    def __add__(self,other):
        other=self.__to_metrics__(other)
        out=Metrics(self.metrics + other.metrics,(self,other))
        def _backward():
            self.grad += Metrics.__fix_dim__(self.grad,out.grad)
            other.grad += Metrics.__fix_dim__(other.grad,out.grad)
        out._backward=_backward
        return out
    def __mul__(self,other):
        other=self.__to_metrics__(other)
        out=Metrics(self.metrics * other.metrics, (self,other))
        def _backward():
            self.grad = Metrics.__fix_dim__(self.grad,other.metrics) * Metrics.__fix_dim__(self.grad,out.grad)
            other.grad = Metrics.__fix_dim__(other.grad,self.metrics) * Metrics.__fix_dim__(other.grad,out.grad)
        out._backward=_backward
        return out
    def __pow__(self,power):
        assert isinstance(power,(int,float)),f"power can't be {type(power)} type"
        out = Metrics(np.power(self.metrics,power),(self,))
        def _backward():
            self.grad += power * np.power(self.metrics,power-1) * Metrics.__fix_dim__(self.grad,out.grad)
        out._backward=_backward
        return out
    def tanh(self):
        exp2=np.exp(self.metrics * 2)
        tanh_metrics=(exp2 - 1)/(exp2 + 1)
        out=Metrics(tanh_metrics,(self,))
        def _backward():
            self.grad += Metrics.__fix_dim__(self.grad,1-out.metrics**2) * Metrics.__fix_dim__(self.grad,out.grad)
        out._backward=_backward
        return out
    def exp(self):
        out = Metrics(np.exp(self.metrics),(self,))
        def _backward():
            self.grad += Metrics.__fix_dim__(self.grad,out.metrics) * Metrics.__fix_dim__(self.grad,out.grad)
        out._backward=_backward
        return out
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
        self.grad=np.ones_like(self.metrics)
        for node in topology:
            node._backward()
