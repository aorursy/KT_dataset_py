import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



x = pd.read_csv('../input/chemical/Logistic_X_Train.csv').values

y = pd.read_csv('../input/chemical/Logistic_Y_Train.csv').values

y = y[:,0]

print(x.shape, y.shape)
def hypothesis(x, w, b):

    h = np.dot(x, w) +b

    return sigmoid(h)



def sigmoid(h):

    return (1.0)/(1.0 + np.exp(1.0*h))



def error(y_true, x, w, b):

    m = x.shape[0]

    err = 0.0

    for i in range(m):

        hx = hypothesis(x[i], w, b)

        err +=y_true[i]*np.log2(hx)  + (1-y_true[i])*np.log2(1-hx)

    return -err/m 

def get_grads(y_true, x, w, b):

    grad_w = np.zeros(w.shape)

    grad_b = 0.0

    m = x.shape[0]

    for i in range(m):

        hx = hypothesis(x[i], w, b)

        grad_w += -1*(y_true[i]-hx)*x[i]

        grad_b = (y_true[i]-hx)

    grad_w /= m

    grad_b /= m

    return grad_w, grad_b

def grad_descent(x, y_true, w, b, rate = 0.5):

    err = error(y_true, x, w, b)

    grad_w, grad_b = get_grads(y_true, x, w, b)

    w = w + rate*grad_w

    b = b + rate*grad_b

    return err, w, b    

def predict(x, w, b):

    c = hypothesis(x, w, b)

    if c <0.5:

        return 0

    else:

        return 1

    

def get_acc (x, w, b):

    y_p = []

    for i in range(x.shape[0]):

        p = predict(x[i], w, b)

        y_p.append(p)

    y_p = np.array(y_p)

    return y_p
loss = []

w= 2*np.random.random((x.shape[1],))

b = 5*np.random.random()

for i in range(100):

    l, w, b = grad_descent(x, y, w, b)

    loss.append(l)

    

x_t = pd.read_csv('../input/chemical/Logistic_X_Test.csv').values

y_t = get_acc(x_t, w, b)

print(x_t.shape)
print(w, b)
df = pd.DataFrame(data = y_t, columns = ["label"])

print(df)

df.to_csv('OUTPUT.csv', index = False)