import numpy as np
from sklearn.metrics import log_loss

pred = np.array([ 0.0734,  0.0677,  0.0628,  0.0731,  0.0818,  0.0757,  0.0919,
        0.101 ,  0.0818,  0.0872,  0.0995,  0.1041])
true = [10]

print(log_loss(true, pred))
from sklearn.preprocessing import LabelBinarizer

def logloss(y, yhat, labels, eps=1e-15):
    """
    y : the true values
    yhat : the predicted values to be measured
    labels : the labels for y
    Ideas taken from: https://github.com/hongguangguo/scikit-learn/blob/4efc12275aadb3ffb21e38137d7e4aeae3e15eba/sklearn/metrics/classification.py
    """
    lb = LabelBinarizer()
    lb.fit(labels)
    y = lb.transform(y)
    yhat = np.clip(yhat, eps, 1 - eps)
    loss = -(y * np.log(yhat)).sum(axis=1)
    loss = np.average(loss)
    return loss

print(logloss(true, pred, list(range(12))))