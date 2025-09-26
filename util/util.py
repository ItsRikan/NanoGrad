import numpy as np

def fix_dim(match_shape:tuple,out:np.ndarray)->np.ndarray:
    out_shape=out.shape
    if match_shape==out_shape:
        return out
    elif np.prod(match_shape)==1:
        return out.sum().reshape(match_shape)
    elif np.prod(out_shape)==1:
        return np.tile(out,match_shape)
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

def set_grad(matrics,out_grad):
        grad =fix_dim(matrics.shape,out_grad)
        try:
            matrics.grad += grad
        except:
            matrics.grad = grad
            return "warning : trying to add none and Matrics"