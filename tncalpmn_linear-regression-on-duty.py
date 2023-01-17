import numpy as np  # Scientific Computation Library

import pandas as pd # Data manupulation

import matplotlib   # For visualisation

import matplotlib.pyplot as plt

from sklearn import linear_model

import sklearn
trainSet = pd.read_csv("../input/train.csv", sep= ',',dtype = np.float16)

testSet = pd.read_csv("../input/test.csv", sep= ',',dtype = np.float16)
print(trainSet.head())

print("Dimensions of trainSet is: ",trainSet.shape)
X = trainSet["x"]

Y = trainSet["y"]



print(X.head(n=3))

print(Y.head(n=3))
trainSet.plot(kind='scatter', x="x",y="y",color='blue', label='JustALabel',s=5)

plt.show()
print("Datatype of trainSet: ",type(trainSet)) # get Information about datatype of variable

print("Datatype of X: ",type(X))        # get Information about datatype of variable

print("Datatype of Y: ",type(Y))        # get Information about datatype of variable

print("X Contains NaN: " , X.isnull().any()) # isnull() function returns an array with boolean type where NaN is 

print("Y Contains NaN: " , Y.isnull().any()) # adding any() will return if array at least one NaN contains
cleanTrainSet = trainSet.dropna() # as name of the function implies, it "drops" all the instances with NaN entry

print(cleanTrainSet.shape)
linearRegressionModel = linear_model.LinearRegression()

#linearRegressionModel.fit(X,Y) # without reshaping arrays, this will throw an error. 

                                # fit function requires numpy array of shape



X=cleanTrainSet.as_matrix(['x'])

Y=cleanTrainSet.as_matrix(['y'])



linearRegressionModel.fit(X,Y)
cleanTrainSet.plot(kind='scatter', x="x",y="y",color='blue', label='JustALabel',s=2)

plt.plot(X, linearRegressionModel.predict(X), color= 'darkred',linewidth=3.0)

cleanTrainSet.plot(kind='hist', color='lightblue', subplots= 'true')

testSet.isnull().any().any()
test_X = testSet.as_matrix(["x"])

test_Y = testSet.as_matrix(["y"])



linRegPredicted = linearRegressionModel.predict(test_X)



matrix = []



for i in range(0,5):

    line = []

    line.append(test_X[i])

    line.append(test_Y[i])

    line.append(linRegPredicted[i])

    matrix.append(line)



dataInfo = pd.DataFrame(matrix, columns=['Variable', 'Original Value','LinearRegression Prediction'])

dataInfo
NNRegression = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5) # Not to be mixed up with KNN classification. Here we are doing regression with labels with no scalar values.

NNRegression.fit(X,Y)

NNRegPredicted = NNRegression.predict(test_X)



matrix2 = []



for i in range(0,10):

    line2 = []

    line2.append(test_X[i])

    line2.append(test_Y[i])

    line2.append(linRegPredicted[i])

    line2.append(NNRegPredicted[i])

    matrix2.append(line2)

    

dataInfo2 = pd.DataFrame(matrix2, columns=['Variable', 'Original Value','LinearRegression Prediction','KNeighborsRegression Prediction'])

dataInfo2