from engine import Matrics
def mean_square_error(y_true:Matrics,y_pred:Matrics):
    loss=(y_true-y_pred)**2
    loss=loss.mean(keepdims=True)
    return loss
