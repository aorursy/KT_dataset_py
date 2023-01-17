from pathlib import Path



DATA_DIR = Path("/kaggle/input")

if (DATA_DIR / "ucfai-core-sp20-rf-svm").exists():

    DATA_DIR /= "ucfai-core-sp20-rf-svm"

else:

    # You'll need to download the data from Kaggle and place it in the `data/`

    #   directory beside this notebook.

    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-rf-svm/data

    DATA_DIR = Path("data")
# Importing the important stuff

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import time

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, matthews_corrcoef

from sklearn.neighbors import KNeighborsClassifier



# change this if running locally

DATA_DIR = "/kaggle/input/ucfai-core-sp20-rf-svm"

# DATA_DIR = "."
from sklearn.datasets import load_iris

iris = load_iris()

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.1)
trees = RandomForestClassifier(n_estimators=150)

trees.fit(X_train, Y_train)
predictions = trees.predict(X_test)

confusion = confusion_matrix(Y_test, predictions)



def plot_confusion_matrix(confusion):

  categories = ["setosa", "versicolor", "virginica"]

  fig, ax = plt.subplots()

  im = ax.imshow(confusion)

  ax.set_yticks(np.arange(len(categories)))

  ax.set_yticklabels(categories)



  for i in range(len(categories)):

    for j in range(len(confusion)):

      ax.text(i, j, confusion[i, j], ha="center", va="center", color="white")



plot_confusion_matrix(confusion)
print(matthews_corrcoef(Y_test, predictions))
# display our training data on a scatter plot



plt.figure(figsize=(12,12))

plt.subplot(441)

plt.scatter(X_train[:,0], X_train[:,1], c=Y_train)

plt.subplot(442)

plt.scatter(X_train[:,1], X_train[:,2], c=Y_train)

plt.subplot(443)

plt.scatter(X_train[:,2], X_train[:,3], c=Y_train)

plt.subplot(444)

plt.scatter(X_train[:,3], X_train[:,0], c=Y_train)
# initialize and train our model

neigh = KNeighborsClassifier(n_neighbors=4)

neigh.fit(X_train, Y_train)
predictions = neigh.predict(X_test)

print("Test Values: ", Y_test)

print("Predictions: ", predictions)





correct = 0

for i in range(0, len(predictions)):

  if predictions[i] == Y_test[i]:

    correct+=1



print("Accuracy: %.2f %%" % (correct * 100 /len(predictions)))
# Ploting predictions

plt.figure(figsize=(12,12))

plt.subplot(441)

plt.scatter(X_test[:,0], X_test[:,1], c=predictions)

plt.subplot(442)

plt.scatter(X_test[:,1], X_test[:,2], c=predictions)

plt.subplot(443)

plt.scatter(X_test[:,2], X_test[:,3], c=predictions)

plt.subplot(444)

plt.scatter(X_test[:,3], X_test[:,0], c=predictions)
# SVM regularization parameter, we'll keep it simple for now

C = 1.0 



models = [svm.SVC(kernel='linear', C=C),

          svm.SVC(kernel='rbf', gamma=0.7, C=C),

          svm.SVC(kernel='poly', degree=3, C=C)]
models = (clf.fit(X_train, Y_train) for clf in models)
results = []



for clf in models:

  predictions = clf.predict(X_test)

  results.append(matthews_corrcoef(Y_test, predictions))

  plot_confusion_matrix(confusion_matrix(Y_test, predictions))

  

print(results)
data = pd.read_csv(f"{DATA_DIR}/train.csv")

data.head()
data.describe()
X = data.drop(labels='Class', axis = 1)

Y = data.loc[:,'Class']



#don't need that original data

del data



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1)



#don't need our X and Y anymore

del X, Y
start = time.time()

# TODO, make the model

# call the model variable: model

# YOUR CODE HERE

raise NotImplementedError()

end = time.time()



# this is going to tell you the time it took to train

print(end-start)
# TODO, use your model to predict for the test set

# needs to be:

# YOUR CODE HERE

raise NotImplementedError()
submission_data = pd.read_csv(f"{DATA_DIR}/test.csv")

predictions = model.predict(submission_data)



predictions = pd.DataFrame({'Id': submission_data[submission_data.columns[0]], 'Class': predictions})



predictions.to_csv('submission.csv', header=['Id', 'Class'], index=False)