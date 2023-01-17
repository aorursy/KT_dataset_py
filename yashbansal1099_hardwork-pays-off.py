#initialize

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read and plot Train Data
x = pd.read_csv('../input/hardwork/Linear_X_Train.csv').values
y = pd.read_csv('../input/hardwork/Linear_Y_Train.csv').values
plt.scatter(x, y)
plt.show()

def hypothesis(x, thetha):
    y_ = thetha[0] + thetha[1]*x
    return y_
def gradient(x, y, thetha):
    m = x.shape[0]
    grad = np.zeros((2, ))
    for i in range(m):
        y_ = hypothesis(x[i], thetha)
        Y = y[i]
        grad[0] += (y_ - Y)
        grad[1] += (y_ - Y)*x[i]
    return grad/m      
def error(x, y, thetha):
    m = x.shape[0]
    total_error = 0.0
    for i in range (m):
        y_ = hypothesis(x[i], thetha)
        total_error += (y_ - y[i])**2
    return total_error/m   
def descent(x, y, steps = 300, rate = 0.03):
    thetha = np.zeros((2,))
    error_list = []
    for i in range(steps):
        grad = gradient(x, y, thetha)
        e = error(x, y, thetha)
        error_list.append(e)
        thetha[0] = thetha[0] - rate*grad[0]
        thetha[1] = thetha[1] - rate*grad[1]
    return thetha, error_list
thetha, error_list = descent(x, y)
plt.plot(error_list)
xt = pd.read_csv('../input/hardwork/Linear_X_Test.csv').values
yt = hypothesis(xt, thetha)
df = pd.DataFrame(data = yt, columns = ["y"])
df.to_csv('Y_Test.csv', index = False)

