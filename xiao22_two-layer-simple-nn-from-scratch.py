import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import OrderedDict
# Import data as Pandas DataFrame

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# Check the train data

train_df.tail()
# Check the test data

test_df.tail()
# Convert datas to numpy array

x_train = train_df.iloc[:,1:].values

t_train_value = train_df.iloc[:,0].values

x_test = test_df.iloc[:,0:].values
# Convert label data to one hot vector structure

t_train = np.zeros((t_train_value.shape[0], 10))

for i in range(t_train.shape[0]):

    t_train[i, t_train_value[i]] = 1
# Define affine layer

class Affine:

    def __init__(self, W, b):

        self.W = W

        self.b = b

    

    def forward(self, x):

        self.x = x

        y = np.dot(x, self.W) + self.b

        return y

    

    def backward(self, dout):

        self.db = np.sum(dout, axis=0)

        self.dW = np.dot(self.x.T, dout)

        dx = np.dot(dout, self.W.T)

        return dx
# Define ReLU layer

class Relu:

    def __init__(self):

        self.mask = None

        pass

    

    def forward(self, x):

        self.mask = (x < 0)

        return np.maximum(0, x)

    

    def backward(self, dout):

        dout[self.mask] = 0

        return dout
# Define softmax function and cross entropy error layer

class SoftmaxWithLoss:

    def __init__(self):

        pass

    

    def forward(self, x, t):

        self.t = t

        

        #softmax

        x = x.T

        x = x - np.max(x)

        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        self.y = y.T

        

        #cross entropy error

        loss = - np.sum(self.t * np.log(self.y + 1e-7)) / self.y.shape[0]

        

        return loss

    

    def backward(self, dout=1):

        return (self.y - self.t) / self.y.shape[0]

        
# Define 2 layer neural network layer

class TwoLayerNet:

    def __init__(self, lr = 0.001, std = 0.01, hidden_size = 50):

        self.lr = lr

        self.W = {}

        

        self.W['W1'] = std * np.random.randn(784, hidden_size)

        self.W['b1'] = np.zeros((1, hidden_size)).astype(float)

        self.W['W2'] = std * np.random.rand(hidden_size, 10)

        self.W['b2'] = np.zeros((1, 10)).astype(float)

        

        self.layers = OrderedDict()

        self.layers['Affine1'] = Affine(self.W['W1'], self.W['b1'])

        self.layers['ReLU1'] = Relu()

        self.layers['Affine2'] = Affine(self.W['W2'], self.W['b2'])

        

        self.last_layer = SoftmaxWithLoss()

        

    def predict(self, x):

        z = x

        for layer in self.layers.values():

            z = layer.forward(z)

        return z

    

    def loss(self, x, t):

        y = self.predict(x)

        loss = self.last_layer.forward(y, t)

        return loss

    

    def accuracy(self, x, t):

        y = self.predict(x)

        y_value = y.argmax(axis=1)

        t_value = t.argmax(axis=1)

        correct = np.sum((y_value == t_value).astype(int))

        return correct / y.shape[0]

    

    def gradient(self, dout=1):

        layers = self.layers

        back_layers = list(reversed(layers))

        

        dout = self.last_layer.backward(dout)

        for layer in back_layers:

            dout = self.layers[layer].backward(dout)

            

        grad = {}

        grad['W1'] = self.layers['Affine1'].dW

        grad['b1'] = self.layers['Affine1'].db

        grad['W2'] = self.layers['Affine2'].dW

        grad['b2'] = self.layers['Affine2'].db

        

        return grad

    

    def learn(self, x, t):

        loss = self.loss(x, t)

        grad = self.gradient()

        

        self.W['W1'] -= self.lr * grad['W1']

        self.W['b1'] -= self.lr * grad['b1']

        self.W['W2'] -= self.lr * grad['W2']

        self.W['b2'] -= self.lr * grad['b2']

        

        return self
net = TwoLayerNet(lr = 0.001, std = 0.01, hidden_size = 100)
# Learn from training data

batch_size = 100

epoch = 100



iteration = int(x_train.shape[0] / batch_size)



for i in range(epoch):

    mask = np.random.choice(x_train.shape[0], x_train.shape[0])

    print('-------------------------------------------------')

    print('Epoch {0}'.format(i+1))

    print('Accuracy: {0}'.format(net.accuracy(x_train, t_train)))

    

    for j in range(iteration):

        x_batch = x_train[mask[j*batch_size:(j+1)*batch_size]]

        t_batch = t_train[mask[j*batch_size:(j+1)*batch_size]]

        net.learn(x_batch, t_batch)
# Predict test data

t_test_prob = net.predict(x_test)

t_test = t_test_prob.argmax(axis=1)
submit_df = pd.DataFrame({'ImageId': np.arange(1, 28001), 'Label': t_test})
submit_df.tail()
submit_df.to_csv('TwoLayerNN.csv', index=False)