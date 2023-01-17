# This is our computation machine.

# It takes an input and constructs and output by multiplying the given input with 4.

def multiply(number):

    return 4 * number



# Let's use it.

output = multiply(3)



# Print the result to screen.

print(output)
# Define the c as a separate global variable to change it easily without redefinig the whole function.

c = 0.5



# First define our machine with the decided values.

def kilometers_to_miles(kilometers):

    miles = kilometers * c

    return miles



# Now evaluate the machine and print the result.

output = kilometers_to_miles(100)

print(output)
# The error function.

def error(truth, calculated):

    return truth - calculated



# Calculate the error of the previous calculation.

output = kilometers_to_miles(100)

err = error(62.137, output)

print("The error is", err)
# Redefine the new c

c = 0.6



output = kilometers_to_miles(100)

err = error(62.137, output)

print("The error is", err)
c = 0.7



output = kilometers_to_miles(100)

err = error(62.137, output)

print("The error is", err)
c = 0.61



output = kilometers_to_miles(100)

err = error(62.137, output)

print("The error is", err)
c = 0.62



output = kilometers_to_miles(100)

err = error(62.137, output)

print("The error is", err)
%matplotlib inline

import matplotlib.pyplot as plt



dataset = {

    "widths": [3.0, 1.0],

    "lengths": [1.0, 3.0],

    "bugs": ["ladybird", "caterpillar"]

}



plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")

plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")

plt.xlabel("Width")

plt.ylabel("Length")

plt.legend()

plt.show()
# To make these calculations easier and some practice purposes let's import numpy here.

import numpy as np



A = 0.25



x = np.linspace(0, 3)

y = A * x



plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")

plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")

plt.plot(x, y, label="y = 0.25 * x")

plt.xlabel("Width")

plt.ylabel("Length")

plt.legend()

plt.show()
y = A * dataset["widths"][0]



err = error(dataset["lengths"][0], y)

print("The error is", err)
y = A * dataset["widths"][0]



err = error(1.1, y)

print("The error is", err)
deltaA = err / dataset["widths"][0]

A = 0.25 + deltaA



y = A * x



plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")

plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")

plt.plot(x, y, label="y = {0:.2} * x".format(A))

plt.xlabel("Width")

plt.ylabel("Length")

plt.legend()

plt.show()
x = dataset["widths"][1]

y = A * x



# We expect to see that 3.0

print(y)
err = error(2.9, y)

print("Error is", err)
# Previous A

A = 0.37

x = np.linspace(0, 3)

deltaA = err / dataset["widths"][1]

A = A + deltaA



y = A * x



plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")

plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")

plt.plot(x, 0.37 * x, label="y = 0.37 * x")

plt.plot(x, y, label="y = {0:.2} * x".format(A))

plt.plot(x, 0.25 * x, label="y = 0.25 * x")

plt.xlabel("Width")

plt.ylabel("Length")

plt.legend()

plt.show()
A = 0.25

L = 0.5

t = np.linspace(0, 3)



x = dataset["widths"][0]

y = A * x

E = error(1.1, y)

print("Error:", E)

deltaA = L * (E / x)

A = A + deltaA



plt.plot(t, A * t, label="First Iteration")

plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")

plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")

plt.xlabel("Width")

plt.ylabel("Length")

plt.legend()

plt.show()
x = dataset["widths"][1]

y = A * x

E = error(0.9, y)

print("Error:", E)

deltaA = L * (E / x)

A = A + deltaA



plt.plot(t, A * t, label="Second Iteration")

plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")

plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")

plt.xlabel("Width")

plt.ylabel("Length")

plt.legend()

plt.show()



print("The final value of A is", A)
def step_function(X):

    return (X > 1.0).astype(int)



x = np.linspace(-10, 10, 1000)

y = step_function(x)



plt.plot(x, y)

plt.title("Step Function")

plt.xlabel("x")

plt.ylabel("y")

plt.show()
def derivative(function, evaluate, h):

    return (function(evaluate + h) - function(evaluate)) / h
plt.figure(figsize=(15,5))



y = derivative(step_function, x, 0.1)

plt.subplot(1, 2, 1)

plt.plot(x, y)

plt.grid(True)



dy = derivative(step_function, x, 0.000001)

plt.subplot(1, 2, 2)

plt.plot(x, dy)

plt.grid(True)

plt.show()
def sigmoid(X):

    return 1 / (1 + np.exp(-X))



plt.figure(figsize=(15,5))



y = sigmoid(x)

plt.subplot(1, 2, 1)

plt.plot(x, y)

plt.title("Sigmoid")

plt.grid(True)



dy = derivative(sigmoid, x, 0.000001)

plt.subplot(1, 2, 2)

plt.plot(x, dy)

plt.title("Derivative of Sigmoid")

plt.grid(True)

plt.show()
w1, w2, w3 = 0.9, 0.3, 0.5

x1, x2, x3 = 1, 2, 0.5



y = sigmoid(x1 * w1 + x2 * w2 + x3 * w3)

print(y)
W = np.array([0.9, 0.3, 0.5])

X = np.array([1, 2, 0.5], ndmin=2).T

y = sigmoid(W.dot(X))

print(y)
W = np.array([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])

I = np.array([0.9, 0.1, 0.8], ndmin=2).T

X_h = W.dot(I)

print(X_h)
O_h = sigmoid(X_h)

print(O_h)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

import scipy.misc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Activation function

def sigmoid(X):

    return 1 / (1 + np.exp(-X))



def ReLU(X):

    return X * (X > 0)
# NN Class

class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, activation=sigmoid):

        self.inodes = inputnodes

        self.hnodes = hiddennodes

        self.onodes = outputnodes

        

        self.lr = learningrate

        

        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5

        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        

        self.activation = activation

    

    def query(self, inputs_list):

        inputs = np.array(inputs_list, ndmin=2).T

        

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation(hidden_inputs)

        

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation(final_inputs)

        

        return final_outputs

    

    def train(self, inputs_list, targets_list):

        inputs = np.array(inputs_list, ndmin=2).T

        targets = np.array(targets_list, ndmin=2).T

        

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation(hidden_inputs)

        

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation(final_inputs)

        

        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)

        

        self.who += self.lr * np.dot((output_errors * scipy.misc.derivative(self.activation, final_inputs, 0.00001)), np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot((hidden_errors * scipy.misc.derivative(self.activation, hidden_inputs, 0.00001)), np.transpose(inputs))

        

        return np.sqrt(np.sum(np.power(output_errors, 2)))
raw_data = pd.read_csv("../input/trainSimple.csv")

raw_data.head(20)
X = raw_data.drop(["A", "B"], axis=1)

y = raw_data[["A", "B"]]



print("First 5 X values:")

print(X.head())



print("First 5 Y values:")

print(y.head())



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_scalar = StandardScaler()

y_scalar = StandardScaler()



X_scalar.fit(X)

y_scalar.fit(y)



X_norm = X_scalar.transform(X_train)

y_norm = y_scalar.transform(y_train)
print("First 5 X values:")

print(X_norm[:5])



print("First 5 Y values:")

print(y_norm[:5])
print(X_norm.shape)

print(y_norm.shape)

#print(X_test.shape)

#print(y_test.shape)
input_nodes = 6

hidden_nodes = 100

output_nodes = 2

learning_rate = 0.9



nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
for count in range(100):

    total = 0

    print("Count:", count)

    for i in range(len(X_norm)):

        total += nn.train(X_norm[i], y_norm[i])

    print("Train error:", total)
error = 0

X_test = X_scalar.transform(X_test)

for i in range(len(X_test)):

    y_hat = nn.query(X_test[i]).T

    y_hat = y_scalar.inverse_transform(y_hat)

    y_real = y_test.iloc[i].values

    error += np.sum(np.power(y_real - y_hat, 2))



print("Error:", error / len(X_test))
