from engine import Matrics
def mean_square_error(y_true:Matrics,y_pred:Matrics):
    loss=(y_true-y_pred)**2
    loss=loss.mean(keepdims=True)
    return loss

def binary_crossentropy(y_true:Matrics,y_pred:Matrics):
    loss= (y_true * y_pred.log()) + ((1-y_true) * (1-y_pred).log())
    loss=loss.mean()
    return loss
