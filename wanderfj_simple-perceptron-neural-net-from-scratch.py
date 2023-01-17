import numpy as np

import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
data = load_breast_cancer()



X,y = data.data,data.target

print(f'X.shape: {X.shape}')

print(f'y.shape: {y.shape}')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(f'X_train.shape: {X_train.shape}')

print(f'X_test.shape: {X_test.shape}')

print(f'y_train.shape: {y_train.shape}')

print(f'y_test.shape: {y_test.shape}')
from sklearn.base import BaseEstimator



class IfesPerceptron(BaseEstimator):

    "Implements a Perceptron-type neural network with a neuron"

    def __init__(self):

        pass



    def funcao_ativacao(self, XWb):

        "The activation function is the step function (also called step or heaviside)"

        # returns 0 se x < 0

        # returns 1 se x >= 0

        return np.heaviside(XWb, 1)



    def fit(self, X, y=None):

        "Train using one instance at a time"

        self.numero_instancias = X.shape[0]

        self.numero_features = X.shape[1]



        # Learning rate

        self.taxa_aprendizado = 0.1



        # W weight matrices

        # According to Tariq Rashid's book, it is good practice to start the weights randomly

        # between +- 1/square_root(nodes_entry)

        self.W = np.random.normal(0.0, pow(self.numero_features, -0.5), (1, self.numero_features))

        print(f'Matrix W Initial ({self.W.shape}): {self.W}')



        # Vetor de peso bias b (inicializada sempre como 1)

        self.b = [1]

        print(f'Vector b: {self.b}')



        for instancia in range(self.numero_instancias):

            XWb = np.dot(X[instancia], self.W.T) + self.b

            h_WbX = self.funcao_ativacao(XWb)

            print(f'XW+b: {XWb}')

            print(f'Result of the activation function: {h_WbX}')



            # Atualização dos pesos de W

            self.W += self.taxa_aprendizado * (y[instancia] - h_WbX) * X[instancia]

            print(f'Matrix W after instance {instancia}: {self.W}')



    def predict(self, X, y=None):

        "Predict all X_test answers using the trained weight matrix"

        XWb = np.dot(X, self.W.T)

        hWbX = self.funcao_ativacao(XWb)

        return hWbX
clf_ifes_perceptron = IfesPerceptron()

clf_ifes_perceptron.fit(X_train,y_train)

y_pred = clf_ifes_perceptron.predict(X_test,y_test)
print(f'ACCURACY SCORE: {accuracy_score(y_pred, y_test)}')