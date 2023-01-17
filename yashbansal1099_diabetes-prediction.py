import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



x = pd.read_csv('../input/diabetes/Diabetes_XTrain.csv').values

y = pd.read_csv('../input/diabetes/Diabetes_YTrain.csv').values.reshape((-1,))

m = np.mean(x, axis = 0)

print (m)

standard= np.std(x, axis = 0)

x = (x-m)/standard
x_t = pd.read_csv('../input/diabetes/Diabetes_Xtest.csv').values

x_t = (x_t - m)/standard
def dist(x1, x2):

    n = len(x1)

    s = 0.0

    for i in range(n):

        d = (x1[i] - x2[i])**2

        s+=d

    return np.sqrt(s)

def knn(x, y, x_t, k = 11):

    vals = []

    m = x.shape[0]

    for i in range(m):

        d = dist(x[i], x_t)

        vals.append((d, y[i]))

    vals.sort()

    

    vals = vals[:k]

    #print(vals)

    #print(m)

    vals = np.array(vals)

    new_vals = np.unique(vals[:, 1], return_counts = True)

    index= new_vals[1].argmax()

    pred = new_vals[0][index]

    #print(pred)

    return pred

    

y_t = []

n = x_t.shape[0]

for i in range(n):

    t = knn(x, y, x_t[i])

    y_t.append(int(t))

y_t = np.array(y_t)

df = pd.DataFrame(data = y_t, columns = ["Outcome"])

print(df)

df.to_csv('OUTPUT.csv', index = False)
