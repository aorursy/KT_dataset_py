import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix



from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
import pandas as pd



location = '../input/iris/Iris.csv'

iris_dataset = pd.read_csv(location)

iris_dataset.head(10)
iris_dataset.hist(alpha=0.5, figsize=(20, 20), color='red')

plt.show()
#separating elements

columns = list(iris_dataset.columns)

X = iris_dataset[[columns[1], columns[2], columns[3], columns[4]]]

y = iris_dataset[columns[5]]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)
def machine_learning_algorithms(X, y):

    train_X = X

    train_y = y

    

    modelForest = RandomForestClassifier(n_estimators=100)

    modelForest.fit( train_X , train_y)

    

    modelLogistic = LogisticRegression()

    modelLogistic.fit(train_X , train_y)

    

    modeltrees = ExtraTreesClassifier(n_estimators=100)

    modeltrees.fit(train_X , train_y)

    

    modelSVC = SVC()

    modelSVC.fit(train_X , train_y)



    modelgbc = GradientBoostingClassifier()

    modelgbc.fit( train_X , train_y)

    

    modelKNN = KNeighborsClassifier(n_neighbors = 3)

    modelKNN.fit( train_X , train_y)

    

    modelGaussian = GaussianNB()

    modelGaussian.fit(train_X , train_y)



    models = [modelForest, modelLogistic, modeltrees, modelSVC, modelgbc, modelKNN, modelGaussian]

    

    return models



np.random.seed(1000)

algorithm = machine_learning_algorithms(X_train, y_train)

columns = ['RandomForest', 'Logistic Regression', 'Tree Classifier', 'SVC', 'GBC', 'KMeans', 'N. Bayes']

scores_train = []

scores_test = []

for k in range(0, 7):

    scores_train.append(algorithm[k].score(X_train, y_train))

    scores_test.append(algorithm[k].score(X_test, y_test))

                

RandomForest = 'Train:{}  Test:{}'.format("%.4f" % scores_train[0], "%.4f" % scores_test[0])

LogisticRegr = 'Train:{}  Test:{}'.format("%.4f" % scores_train[1], "%.4f" % scores_test[1])

TreeClassifier = 'Train:{}  Test:{}'.format("%.4f" % scores_train[2], "%.4f" % scores_test[2])

SVC = 'Train:{}  Test:{}'.format("%.4f" % scores_train[3], "%.4f" % scores_test[3])

GBC = 'Train:{}  Test:{}'.format("%.4f" % scores_train[4], "%.4f" % scores_test[4])

Kmeans = 'Train:{}  Test:{}'.format("%.4f" % scores_train[5], "%.4f" % scores_test[5])

NBayes = 'Train:{}  Test:{}'.format("%.4f" % scores_train[6], "%.4f" % scores_test[6])
print("Random Forest                     :", RandomForest) 

print("Logistic Regression               :", LogisticRegr)

print("Tree Classifier                   :", TreeClassifier)

print("Support-Vector Machine(SVM)       :", SVC)

print("Gradient Boosting Classifier (GBC):", GBC)

print("KMeans                            :", Kmeans)

print("Naive Bayes                       :", NBayes)
plt.figure(figsize=(15,6))

plt.title(' Different model comparison: Train Score')

plt.ylabel(' Score')

bar = plt.bar(columns, scores_train)

bar[0].set_color('red')

bar[1].set_color('green')

bar[2].set_color('blue')

bar[3].set_color('pink')

bar[4].set_color('orange')

bar[5].set_color('gray')

bar[6].set_color('black')
plt.figure(figsize=(15,6))

plt.title(' Different model comparison: Test Score')

plt.ylabel(' Score')

bar = plt.bar(columns, scores_test)

bar[0].set_color('red')

bar[1].set_color('green')

bar[2].set_color('blue')

bar[3].set_color('pink')

bar[4].set_color('orange')

bar[5].set_color('gray')

bar[6].set_color('black')