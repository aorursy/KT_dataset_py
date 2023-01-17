import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
train = pd.read_csv("../input/heart-disease/heart.csv")
X = train.iloc[:, :-1]
y = train.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.9)
print(y_train.shape)
print(X_train.shape)
reg = RandomForestClassifier()
reg.fit(X_train, y_train)
# reg.predict(X_test)
reg.score(X_test, y_test)
y_pred = reg.predict(X_test)
y_pred
df = pd.DataFrame(data = y_pred, columns=['Prediction'])
df.to_csv("Solution.csv")
