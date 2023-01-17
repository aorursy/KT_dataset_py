import numpy as np
import pandas as pd
# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error

y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
# 0.55317
# RMSLE (Root Mean Squared Logarithmic Error)
from sklearn.metrics import mean_squared_log_error

y_true = [100, 0, 400]
y_pred = [200, 10, 200]

rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
print(rmsle)
# 1.49449
# MAE (Mean Absolute Error)
from sklearn.metrics import mean_absolute_error

y_true = [100, 160, 60]
y_pred = [80, 100, 100]

mae = mean_absolute_error(y_true, y_pred)
print(mae)
# 40.0
# accuracy, error rate
from sklearn.metrics import accuracy_score

# Binary classification of 0 and 1
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
# 0.625
# precision, recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Binary classification of 0 and 1
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(precision, recall)
# 0.75 0.6
# logloss
from sklearn.metrics import log_loss

# True value and predicted probability of binary classification of 0 and 1
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)
# 0.71356
# confusion matrix
from sklearn.metrics import confusion_matrix

# Binary classification of 0 and 1
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]


# TP(True Positive), TN(True Negative), FP(False Positive), FN(False Negative)
tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp], [fn, tn]])
print(confusion_matrix1)

# array([[TP, FP]
#        [FN, TN]])


confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)

# array([[TN, FP]
#        [FN, TP]]) 
