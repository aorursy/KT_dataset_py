import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def load_data(train_row):
    train = pd.read_csv('./train.csv')
    X_train = train.values[0:train_row, 1:]
    y_train = train.values[0:train_row, 0]
    
    pred_test = pd.read_csv('./test.csv').values
    
    return X_train, y_train, pred_test

train_row = 5000
Org_X_train, Org_y_train, pred_test = load_data(train_row)
row = 4
print(Org_y_train[row])
plt.imshow(Org_X_train[row].reshape(28, 28))
plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Org_X_train, Org_y_train, test_size = 0.2, random_state = 0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

k_range = range(1, 10)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    scores.append(accuracy)
    print('k is ' + str(k) + ' accuracy is ' + str(accuracy))
