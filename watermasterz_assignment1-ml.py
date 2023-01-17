import numpy as np

import numpy as np

import matplotlib.pyplot as plt
Num_points = 150
x = np.linspace(0, 20, Num_points)

y = np.sin(x)

noise = np.random.normal(0, 0.4, Num_points)
plt.figure(figsize=(15, 7))

plt.xlabel('X')

plt.ylabel('Y')

plt.scatter(x, y+noise , label='Y with gaussian noise')

plt.scatter(x, y, label='pure Y')

plt.legend();
y = y+noise
ones = np.zeros(x.shape)+1
def make_poly_regressor(degree):

    xtrain = np.zeros(x.shape) +1



    for i in range(1, degree, 1):

        xtrain = np.c_[xtrain, x**i]



    xTx_inv = np.linalg.inv(xtrain.T.dot(xtrain))

    #print("X transpose X shape: ",xTx_inv.shape)



    xTy = xtrain.T.dot(y)

    #print("X transpose y shape: ", xTy.shape)



    weights = xTx_inv.dot(xTy)

    #print("Weights shape: ",weights.shape)



    preds = xtrain.dot(weights)

    error = np.mean(np.abs(preds - y))

    plt.plot(x, preds, linewidth= int(0.6/(error**2) ), label = f'Degree: {degree}  error: {error:.2f}')

    
done = 1

plt.figure(figsize=(25, 15))

plt.xlabel("X")

plt.ylabel("Y")

plt.ylim((-4,4))

for i in range(3, 16, 1):

    if done:

        plt.scatter(x,y, label='Y')

        done = 0

    make_poly_regressor(i)

    plt.legend()