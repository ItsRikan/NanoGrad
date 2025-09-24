from engine import Matrics

def to_matrics(matrics):
    if isinstance(matrics,Matrics):
        return matrics
    elif isinstance(matrics,(np.ndarray,list)):
        return Matrics(matrics)
    elif isinstance(matrics,(int,float)):
        return Matrics([matrics])
    else:
        raise TypeError("must be list,numpy array or matrics")
