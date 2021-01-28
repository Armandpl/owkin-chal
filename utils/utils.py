import numpy as np
from sklearn.metrics import confusion_matrix

def get_accuracy(y_true, y_pred):
    assert y_true.ndim == 1 and y_true.size() == y_pred.size()
    y_pred = y_pred > 0.5
    return (y_true == y_pred).sum().item() / y_true.size(0)

def get_accuracy_per_class(y_true, y_pred):
    y_pred = y_pred > 0.5
    #Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    #Now the normalize the diagonal entries
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #The diagonal entries are the accuracies of each class
    return cm.diagonal()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
