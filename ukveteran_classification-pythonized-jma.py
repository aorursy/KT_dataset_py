import pandas as pd



data = pd.read_csv('../input/vertebrate-data/vertebrate.csv',header='infer')

data
data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')

data
pd.crosstab([data['Warm-blooded'],data['Gives Birth']],data['Class'])
from sklearn import tree



Y = data['Class']

X = data.drop(['Name','Class'],axis=1)



clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)

clf = clf.fit(X, Y)
testData = [['gila monster',0,0,0,0,1,1,'non-mammals'],

           ['platypus',1,0,0,0,1,1,'mammals'],

           ['owl',1,0,0,1,1,0,'non-mammals'],

           ['dolphin',1,1,1,0,0,0,'mammals']]

testData = pd.DataFrame(testData, columns=data.columns)

testData
testY = testData['Class']

testX = testData.drop(['Name','Class'],axis=1)



predY = clf.predict(testX)

predictions = pd.concat([testData['Name'],pd.Series(predY,name='Predicted Class')], axis=1)

predictions
from sklearn.metrics import accuracy_score



print('Accuracy on test data is %.2f' % (accuracy_score(testY, predY)))
import numpy as np

import matplotlib.pyplot as plt

from numpy.random import random



%matplotlib inline



N = 1500



mean1 = [6, 14]

mean2 = [10, 6]

mean3 = [14, 14]

cov = [[3.5, 0], [0, 3.5]]  # diagonal covariance



np.random.seed(50)

X = np.random.multivariate_normal(mean1, cov, int(N/6))

X = np.concatenate((X, np.random.multivariate_normal(mean2, cov, int(N/6))))

X = np.concatenate((X, np.random.multivariate_normal(mean3, cov, int(N/6))))

X = np.concatenate((X, 20*np.random.rand(int(N/2),2)))

Y = np.concatenate((np.ones(int(N/2)),np.zeros(int(N/2))))



plt.plot(X[:int(N/2),0],X[:int(N/2),1],'r+',X[int(N/2):,0],X[int(N/2):,1],'k.',ms=4)
# Training and Test set creation



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=1)



from sklearn import tree

from sklearn.metrics import accuracy_score



# Model fitting and evaluation



maxdepths = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]



trainAcc = np.zeros(len(maxdepths))

testAcc = np.zeros(len(maxdepths))



index = 0

for depth in maxdepths:

    clf = tree.DecisionTreeClassifier(max_depth=depth)

    clf = clf.fit(X_train, Y_train)

    Y_predTrain = clf.predict(X_train)

    Y_predTest = clf.predict(X_test)

    trainAcc[index] = accuracy_score(Y_train, Y_predTrain)

    testAcc[index] = accuracy_score(Y_test, Y_predTest)

    index += 1

    

# Plot of training and test accuracies

    

plt.plot(maxdepths,trainAcc,'ro-',maxdepths,testAcc,'bv--')

plt.legend(['Training Accuracy','Test Accuracy'])

plt.xlabel('Max depth')

plt.ylabel('Accuracy')
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

%matplotlib inline



numNeighbors = [1, 5, 10, 15, 20, 25, 30]

trainAcc = []

testAcc = []



for k in numNeighbors:

    clf = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)

    clf.fit(X_train, Y_train)

    Y_predTrain = clf.predict(X_train)

    Y_predTest = clf.predict(X_test)

    trainAcc.append(accuracy_score(Y_train, Y_predTrain))

    testAcc.append(accuracy_score(Y_test, Y_predTest))



plt.plot(numNeighbors, trainAcc, 'ro-', numNeighbors, testAcc,'bv--')

plt.legend(['Training Accuracy','Test Accuracy'])

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')
from sklearn import linear_model

from sklearn.svm import SVC



C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]

LRtrainAcc = []

LRtestAcc = []

SVMtrainAcc = []

SVMtestAcc = []



for param in C:

    clf = linear_model.LogisticRegression(C=param)

    clf.fit(X_train, Y_train)

    Y_predTrain = clf.predict(X_train)

    Y_predTest = clf.predict(X_test)

    LRtrainAcc.append(accuracy_score(Y_train, Y_predTrain))

    LRtestAcc.append(accuracy_score(Y_test, Y_predTest))



    clf = SVC(C=param,kernel='linear')

    clf.fit(X_train, Y_train)

    Y_predTrain = clf.predict(X_train)

    Y_predTest = clf.predict(X_test)

    SVMtrainAcc.append(accuracy_score(Y_train, Y_predTrain))

    SVMtestAcc.append(accuracy_score(Y_test, Y_predTest))



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

ax1.plot(C, LRtrainAcc, 'ro-', C, LRtestAcc,'bv--')

ax1.legend(['Training Accuracy','Test Accuracy'])

ax1.set_xlabel('C')

ax1.set_xscale('log')

ax1.set_ylabel('Accuracy')



ax2.plot(C, SVMtrainAcc, 'ro-', C, SVMtestAcc,'bv--')

ax2.legend(['Training Accuracy','Test Accuracy'])

ax2.set_xlabel('C')

ax2.set_xscale('log')

ax2.set_ylabel('Accuracy')
from sklearn.svm import SVC



C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]

SVMtrainAcc = []

SVMtestAcc = []



for param in C:

    clf = SVC(C=param,kernel='rbf',gamma='auto')

    clf.fit(X_train, Y_train)

    Y_predTrain = clf.predict(X_train)

    Y_predTest = clf.predict(X_test)

    SVMtrainAcc.append(accuracy_score(Y_train, Y_predTrain))

    SVMtestAcc.append(accuracy_score(Y_test, Y_predTest))



plt.plot(C, SVMtrainAcc, 'ro-', C, SVMtestAcc,'bv--')

plt.legend(['Training Accuracy','Test Accuracy'])

plt.xlabel('C')

plt.xscale('log')

plt.ylabel('Accuracy')
from sklearn import ensemble

from sklearn.tree import DecisionTreeClassifier



numBaseClassifiers = 500

maxdepth = 10

trainAcc = []

testAcc = []



clf = ensemble.RandomForestClassifier(n_estimators=numBaseClassifiers)

clf.fit(X_train, Y_train)

Y_predTrain = clf.predict(X_train)

Y_predTest = clf.predict(X_test)

trainAcc.append(accuracy_score(Y_train, Y_predTrain))

testAcc.append(accuracy_score(Y_test, Y_predTest))



clf = ensemble.BaggingClassifier(DecisionTreeClassifier(max_depth=maxdepth),n_estimators=numBaseClassifiers)

clf.fit(X_train, Y_train)

Y_predTrain = clf.predict(X_train)

Y_predTest = clf.predict(X_test)

trainAcc.append(accuracy_score(Y_train, Y_predTrain))

testAcc.append(accuracy_score(Y_test, Y_predTest))



clf = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=maxdepth),n_estimators=numBaseClassifiers)

clf.fit(X_train, Y_train)

Y_predTrain = clf.predict(X_train)

Y_predTest = clf.predict(X_test)

trainAcc.append(accuracy_score(Y_train, Y_predTrain))

testAcc.append(accuracy_score(Y_test, Y_predTest))



methods = ['Random Forest', 'Bagging', 'AdaBoost']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

ax1.bar([1.5,2.5,3.5], trainAcc)

ax1.set_xticks([1.5,2.5,3.5])

ax1.set_xticklabels(methods)

ax2.bar([1.5,2.5,3.5], testAcc)

ax2.set_xticks([1.5,2.5,3.5])

ax2.set_xticklabels(methods)