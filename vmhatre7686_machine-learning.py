import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import sparse


#create a 2D NumPy array with a diagonal of ones and zeros everywhere elese
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

import matplotlib.pyplot as plt
#Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
#Create a second array using sine
y=np.sin(x)
#Plot function makes a line chart of one array against another
plt.plot(x, y, marker="x")
import pandas as pd
# create a simple dataset of people
data = {'Name':["John", "Anna", "Peter","Linda"],
        'Location': ["New York", "Paris", "Berlin", "London"],
        'Age' :[24, 13, 53, 33]}
data_pandas = pd.DataFrame(data)
display(data_pandas)

#Select all rows that have an age column greater than 30
display(data_pandas[data_pandas.Age > 30])


from sklearn.datasets import load_iris
iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
#DESCR value
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of Data: {}".format(type(iris_dataset['data'])))
print("Shape of Data: {}".format(iris_dataset['data'].shape))
print("First Five Columns of Data \n{}".format(iris_dataset['data'][:5]))
#Target one dimensinal array with one entry per flower
print("Shape of target: {}".format(iris_dataset['target'].shape))
#Species encoded as integers from 0 to 2
print("Target: \n {}".format(iris_dataset['target']))
# 0 means setosa, 1 means versicolor, 2 means virginica

#In scikit-learn there is a function, train_test_split, which shuffles the dataset and splits it for you.
#The function extracts 75% of the data as the training set and the remaining 25% is the test set
#! pip install mglearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

#Creating Pair Plot to examine all possibile pairs of fetures.
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
#pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
#                          hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Making Predictions - sample iris in the wild measurements - 
# [sepal length, sepal width, petal length, petal width]
X_new = np.array([[3, 2, .5, 0.1]])
print("X_new.shape: {}".format(X_new.shape))

# To make a prediction we use predicton method
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
    iris_dataset['target_names'][prediction]))

#Evaluating Model
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

#Accuracy of Model
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

#Accuracy of knn object
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))