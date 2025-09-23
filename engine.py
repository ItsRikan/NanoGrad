import numpy as np 

class Matrics:
    def __init__(self,matrics,child=tuple()):
        self.matrics=np.array(matrics,dtype=np.float32)
        #assert self.matrics.ndim==2,'only two dimension matrics is acceptable'
        self.grad=None
        self.shape=self.matrics.shape
        self._backward= lambda:None
        self.child=child
    def __to_matrics__(self,matrics):
        if isinstance(matrics,Matrics):
            return matrics
        elif isinstance(matrics,(np.ndarray,list)):
            return Matrics(matrics)
        elif isinstance(matrics,(int,float)):
            return Matrics([matrics])
        else:
            raise TypeError("must be list,numpy array or matrics")

    def to_list(self):
        return self.matrics.tolist()

    @staticmethod
    def __fix_dim__(match_shape:tuple,out:np.ndarray)->np.ndarray:
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
            elif out_shape[0]==1 and out_shape[1]==1:
                return np.tile(out,match_shape)
            elif out_shape[1]==match_shape[1] and out_shape[0]==1:
                return np.tile(out,(match_shape[0],1))
            elif out_shape[0]==match_shape[0] and out_shape[1]==1:
                return np.tile(out,(1,match_shape[1]))
    @staticmethod
    def __set_grad__(matrics,out_grad):
        try:
            matrics.grad += Matrics.__fix_dim__(matrics.shape,out_grad)
        except:
            matrics.grad = Matrics.__fix_dim__(matrics.shape,out_grad)
            return "warning : trying to add none and Matrics"
    def __add__(self,other):
        other=self.__to_matrics__(other)
        out=Matrics(self.matrics + other.matrics,(self,other))
        def _backward():
            Matrics.__set_grad__(self,out.grad)
            Matrics.__set_grad__(other,out.grad)
            # self.grad += Matrics.__fix_dim__(self.shape,out.grad)
            # other.grad += Matrics.__fix_dim__(other.shape,out.grad)
        out._backward=_backward
        return out
    def __mul__(self,other):
        other=self.__to_matrics__(other)
        out=Matrics(self.matrics * other.matrics, (self,other))
        def _backward():
            Matrics.__set_grad__(self,other.matrics * out.grad)
            Matrics.__set_grad__(other,self.matrics * out.grad)
            # self.grad += Matrics.__fix_dim__(self.shape,other.matrics * out.grad)
            # other.grad += Matrics.__fix_dim__(other.shape,self.matrics * out.grad)
        out._backward=_backward
        return out
    def mean(self,dim:int=None,keepdims=False):
        out=Matrics(self.matrics.mean(dim,keepdims=keepdims),(self,))
        def _backward():
            Matrics.__set_grad__(self,(1/np.prod(self.shape) * out.grad))
            # self.grad += Matrics.__fix_dim__(self.shape,(1/np.prod(self.shape) * out.grad))
        out._backward = _backward
        return out
    def __matmul__(self,other):
        other=self.__to_matrics__(other)
        out = Matrics((self.matrics @ other.matrics),(self,other))
        def _backward():
            Matrics.__set_grad__(self,(out.grad @ other.matrics.T))
            Matrics.__set_grad__(other,(self.matrics.T @ out.grad))
            # self.grad += Matrics.__fix_dim__(self.shape,(out.grad @ other.matrics.T))
            # other.grad += Matrics.__fix_dim__(other.shape,(self.matrics.T @ out.grad))
        out._backward=_backward
        return out
    def __pow__(self,power):
        assert isinstance(power,(int,float)),f"power can't be {type(power)} type"
        out = Matrics(np.power(self.matrics,power),(self,))
        def _backward():
            Matrics.__set_grad__(self,power * np.power(self.matrics,power-1) * out.grad)
            # self.grad += power * np.power(self.matrics,power-1) * Matrics.__fix_dim__(self.shape,out.grad)
        out._backward=_backward
        return out
    def tanh(self):
        tanh_matrics= np.tanh(self.matrics)
        out=Matrics(tanh_matrics,(self,))
        def _backward():
            Matrics.__set_grad__(self,(1-out.matrics**2) * out.grad)
            # self.grad += Matrics.__fix_dim__(self.shape,(1-out.matrics**2) * out.grad)
        out._backward=_backward
        return out
    def sigmoid(self):
        neg_exp=np.exp(-1 * self.matrics)
        sig_matrics=1/(1+neg_exp)
        out=Matrics(sig_matrics,(self,))
        def _backward():
            Matrics.__set_grad__(self,sig_matrics * (1-sig_matrics) * out.grad)
            # self.grad += Matrics.__fix_dim__(self.shape, sig_matrics * (1-sig_matrics) * out.grad)
        out._backward=_backward
        return out
    # def softmax(self,dim:int):
    #     exp=np.exp(self.matrics)
    #     s=exp.sum(dim,keepdims=True)
    #     soft_matrics=exp/s
    #     out=Matrics(soft_matrics,(self,))
    #     def _backward():
    #         self.grad += Matrics.__fix_dim__(self.grad,)
    #     out._backward=_backward
    #     return out
               
    def ReLU(self):
        relu_matrics=np.maximum(self.matrics,0)
        out=Matrics(relu_matrics,(self,))
        def _backward():
            Matrics.__set_grad__(self,np.minimum(relu_matrics,1) * out.grad)
            # self.grad += Matrics.__fix_dim__(self.shape,np.minimum(relu_matrics,1) * out.grad)
    def exp(self):
        out = Matrics(np.exp(self.matrics),(self,))
        def _backward():
            Matrics.__set_grad__(self,out.matrics * out.grad)
            # self.grad += Matrics.__fix_dim__(self.shape,out.matrics * out.grad)
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