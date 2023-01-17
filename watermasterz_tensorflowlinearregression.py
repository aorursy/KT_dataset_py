import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt
dir = '/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv'
# Equivalent to np.array()

data = tf.Variable(pd.read_csv(dir))



print(data.shape)



X = data[:, 0]

Y = data[:, 1]



# Equivalent to np.mean()

means = tf.reduce_mean(Y)



# Scale the data down so that it doesnt cause any problems later

Y = Y/means
plt.figure(figsize=(15, 7))

plt.scatter(X, Y*means, color='red')

plt.xlabel("Years of Experiance")

plt.ylabel("Salary")

plt.legend()
import numpy as np

class LinReg1:

    def __init__(self):

        self.num=0

        self.den=1

        self.m = 0

        self.c = 0

    

    def fit(self, X, Y):

        

        # equivalent to np.sum((X - tf.mean(X))*(Y - tf.mean(Y)))

        self.num = tf.reduce_sum((X - tf.reduce_mean(X))*(Y - tf.reduce_mean(Y)))

        self.den = tf.reduce_sum((X - tf.reduce_mean(X))**2)

        

        self.m = self.num/self.den

        self.c = tf.reduce_sum(tf.reduce_mean(Y) - tf.reduce_mean(X) * self.m)

    

    def predict(self,X):



        return (X*self.m + self.c)

    

    def evaluate(self, X, Y):

        ypreds = self.predict(X)

        error = tf.square(ypreds-Y)

        return tf.reduce_mean(error)



    def plot(self, X, Y):

        preds = self.predict(X)

        plt.figure(figsize=(15,7))

        # Multiply back with mean of the real data to get correct prediction scale 

        plt.plot(X, preds * means, color='red', label='Regressor Line')

        plt.scatter(X, Y * means, color='green', label='Targets')

        plt.legend()

lin = LinReg1()
lin.fit(X, Y)

print(f"Slope: {lin.m}\nIntercept: {lin.c}")

print(f"Mean squared error: {lin.evaluate(X, Y)}")

lin.plot(X, Y)