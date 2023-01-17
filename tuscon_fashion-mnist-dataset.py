# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import struct
import matplotlib.pyplot as plt
import random

def load_data():
    with open('/kaggle/input/fashionmnist/train-labels-idx1-ubyte', 'rb') as labels:
        magic, n =struct.unpack('>II', labels.read(8))
        train_labels=np.fromfile(labels, dtype=np.uint8)
    with open('/kaggle/input/fashionmnist/train-images-idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols=struct.unpack('>IIII', imgs.read(16))
        train_images=np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)

    with open('/kaggle/input/fashionmnist/t10k-labels-idx1-ubyte', 'rb') as labels:
        magic, n =struct.unpack('>II', labels.read(8))
        test_labels=np.fromfile(labels, dtype=np.uint8)
    with open('/kaggle/input/fashionmnist/t10k-images-idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols=struct.unpack('>IIII', imgs.read(16))
        test_images=np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)

    return train_images, train_labels, test_images, test_labels


train_x, train_y, test_x, test_y=load_data()
def visualization_data(img_array, label_array):
    fig, ax= plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)
    ax=ax.flatten()
    for i in range(16):
        img=img_array[label_array==8][i].reshape(28,28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()
visualization_data(train_x, train_y)
def one_hot_enc(y, num_labels=10):
    one_hot = np.zeros((num_labels, y.shape[0]))
    for i, val in enumerate(y):
        one_hot[val][i] = 1.0
    return one_hot
# This is the Activation function for our neural network model.
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

# This is required in backward propogation in our neural network.
def sigmoid_gradient(z):
    s=sigmoid(z)
    return s*(1-s)
def calc_cost(y_enc, output):
    t1= -y_enc*np.log(output)
    t2= (1-y_enc)*(1-np.log(1-output))
    cost=np.sum(t1-t2)
    return cost
def add_bias_unit(X, where):
    if where == 'column':
        X_new=np.ones((X.shape[0], X.shape[1] + 1))
        # print(X_new)
        X_new[:,1:]=X
    elif where == 'row':
        X_new=np.ones((X.shape[0] + 1, X.shape[1]))
        # print(X_new)
        X_new[1:, :]=X

    return X_new
def weights (n_features,hidden_layer,output):
    W1 = np.random.random(size=hidden_layer * (n_features + 1))
    W1 = W1.reshape(hidden_layer, n_features + 1)
    W2 = np.random.random(size=hidden_layer * (hidden_layer + 1))
    W2 = W2.reshape(hidden_layer, hidden_layer + 1)
    W3 = np.random.random(size=output * (hidden_layer + 1))
    W3 = W3.reshape(output, hidden_layer + 1)
    return W1,W2,W3
def feed_forward(x, w1, w2, w3):
    # Add bais to our neural network because column within  the row is just a byte of data so we have to add bais at columns 
   
    a1=add_bias_unit(x, where='column') # This is the input of our model which is train_x
    z2=w1.dot(a1.T)
    a2=sigmoid(z2)
    #Since we transposed we have to add bias in to the row
    a2=add_bias_unit(a2, where='row')
    z3=w2.dot(a2)
    a3=sigmoid(z3)

    a3=add_bias_unit(a3, where='row')
    z4=w3.dot(a3)
    a4=sigmoid(z4)

    return a1, z2, a2, z3, a3, z4, a4
def predict(x, w1, w2, w3):
    a1, z2, a2, z3, a3, z4, a4=feed_forward(x, w1, w2, w3)
    y_pred=np.argmax(a4, axis=0)
    return y_pred
def back_ward(a1, a2, a3, a4, z2, z3, z4, y_enc, w1, w2, w3):
    delta4=a4-y_enc
    
    z3=add_bias_unit(z3, where='row')
    delta3=w3.T.dot(delta4)*sigmoid_gradient(z3)
    delta3=delta3[1:,:]
    
    z2=add_bias_unit(z2, where='row')
    delta2=w2.T.dot(delta3)*sigmoid_gradient(z2)
    delta2=delta2[1:,:]

    grad1=delta2.dot(a1)
    grad2=delta3.dot(a2.T)
    grad3=delta4.dot(a3.T)

    return grad1, grad2, grad3
def runmodel(X,y, X_t, y_t):
    
    #Shuffle, epochs and batch
    
    X_copy, y_copy=X.copy(), y.copy()
    y_enc=one_hot_enc(y)
    print(f"One Hot Encoding Matrix : {y_enc}")
    epochs=100

    batch=50
    w1,w2,w3 = weights(784,75,10)
    # Hyper Parameter are alpha, eta and dec .We only use alpha for now if required then we use eta and dec also.
    
    alpha=0.0001 # step size
    delta_w1_prev = np.zeros(w1.shape)
    delta_w2_prev = np.zeros(w2.shape)
    delta_w3_prev = np.zeros(w3.shape)
    total_cost = []

    pred_acc=np.zeros(epochs)
    for i in range(epochs):

        shuffle = np.random.permutation(y_copy.shape[0])
        X_copy, y_enc=X_copy[shuffle], y_enc[:,shuffle]
        

        mini=np.array_split(range(y_copy.shape[0]), batch)

        for step in mini:
            #Feed Forward
            a1, z2, a2, z3, a3, z4, a4=feed_forward(X_copy[step], w1, w2, w3)
            cost=calc_cost(y_enc[:, step], a4)

            total_cost.append(cost)

            #Backward Propogation
            grad1, grad2, grad3=back_ward(a1,a2,a3,a4,z2,z3,z4,y_enc[:,step],
                                          w1,w2,w3)
            delta_w1, delta_w2, delta_w3 = grad1, grad2, grad3

            w1 -= delta_w1 + alpha*delta_w1_prev
            w2 -= delta_w2 + alpha*delta_w2_prev
            w3 -= delta_w3 + alpha*delta_w3_prev

            delta_w1_prev, delta_w2_prev, delta_w3_prev = delta_w1, delta_w2, delta_w3


        y_pred=predict(X_t, w1, w2, w3)
        acc=np.sum(y_t==y_pred, axis=0)/X_t.shape[0]
        pred_acc[i]=100*np.sum(y_t==y_pred, axis=0)/X_t.shape[0]
        print('epochs #', i)
    print(f"Predicted Value using Model : {pred_acc}")
    return total_cost, pred_acc, y_pred

train_x, train_y, test_x, test_y=load_data()
cost, acc, y_pred=runmodel(train_x, train_y, test_x, test_y)
miscl_img=test_x[test_y != y_pred][:9]
correct_lab=test_y[test_y != y_pred][:9]
miscl_lab=y_pred[test_y != y_pred][:9]
fig, ax= plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
ax=ax.flatten()
plt.title("Prediction of MNIST DataSet")
for i in range(9):
    img=miscl_img[i].reshape(28,28)
    ax[i].imshow(img, cmap='twilight_shifted', interpolation='nearest')
    ax[i].set_title('%d) Label:  %d p: %d' %(i + 1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()

plt.show()
