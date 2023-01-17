import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB
mydata = pd.read_csv('../input/Bank_Personal_Loan_Modelling.csv')

mydata.head()
mydata.shape
mydata.info()
print(mydata.isnull().values.any())
mydata.groupby('Personal Loan').count()
mydata.describe().transpose()
mydata.hist(column=['CCAvg','Age','Personal Loan','Income'],figsize=(20,20))
mydata.skew()
for column in mydata[['Age','Income','Personal Loan']]:

    val = column

    q1 = mydata[val].quantile(0.25)

    q3 = mydata[val].quantile(0.75)

    iqr = q3-q1

    fence_low  = q1-(1.5*iqr)

    fence_high = q3+(1.5*iqr)

    df_out = mydata.loc[(mydata[val] < fence_low) | (mydata[val] > fence_high)]

    if df_out.empty:

        print('No Outliers in the ' + val + ' column of given dataset')

    else:

        print('There are Outliers in the ' + val + ' column of given dataset')
plt.figure(figsize = (15,10))

sns.heatmap(mydata.corr(), annot = True )
sns.scatterplot(y = 'Income', x = 'Age', data = mydata, hue = 'Personal Loan')
sns.scatterplot(x = 'Income', y = 'CCAvg', data = mydata, hue = 'Personal Loan')
sns.scatterplot(y = 'Mortgage', x = 'Income', data = mydata, hue = 'Personal Loan')
sns.scatterplot(y = 'Family', x = 'Income', data = mydata, hue = 'Personal Loan')
mydata = mydata.drop(['ID','Experience', 'ZIP Code', 'Age', 'Online', 'CreditCard', 'Securities Account'], axis = 1)
mydata.head()
mydata[['Income', 'CCAvg', 'Mortgage']] = mydata[['Income', 'CCAvg', 'Mortgage']] * 1000
mydata.head()
a = mydata.drop('Personal Loan', axis = 1)

b = mydata['Personal Loan']
a_train, a_test, b_train, b_test = train_test_split(a,b, test_size = 0.30, random_state = 1)



trainingData = preprocessing.scale(a_train)

testData = preprocessing.scale(a_test)



logReg = LogisticRegression()

logReg.fit(trainingData, b_train)

pred = logReg.predict(testData)

logReg_score = logReg.score(testData, b_test)

print('logReg_score: ' , logReg_score)

accuracy = logReg.score(testData, b_test) * 100

accuracies = {}

accuracies['LR'] = accuracy

print('Logistic Regression Accuracy: ', accuracy)

print('Confusion Matrix: ')

print(metrics.confusion_matrix(b_test, pred))

nb = GaussianNB()

nb.fit(trainingData, b_train)

expected  = b_test

predicted = nb.predict(testData)

print(metrics.classification_report(expected, predicted))

accuracy = (metrics.accuracy_score(expected, predicted))*100

accuracies['NB'] = accuracy

print('Naive Bayes accuracy: ', accuracy)

print('Confusion Matrix: ')

print(metrics.confusion_matrix(expected, predicted))
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore



knn = KNeighborsClassifier(n_neighbors = 2, weights = 'distance')

c = a.apply(zscore)

c.describe()
a = np.array(c)

print(a.shape)

b = np.array(b)

print(b.shape)
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.30, random_state=1)

knn.fit(a_train, b_train)

prediction = knn.predict(a_test)

knn.score(a_test, b_test)

accuracy = metrics.accuracy_score(b_test, prediction)*100

print(accuracy)

accuracies['KNN'] = accuracy

print('KNN accuracy: ', accuracy)

print('Confusion Matrix: ')

print(metrics.confusion_matrix(b_test, prediction))
from sklearn import svm



svm = svm.SVC(gamma = 0.025, C = 3)

svm.fit(trainingData, b_train)

svm.score(testData, b_test)

predicted = svm.predict(testData)

accuracy = (metrics.accuracy_score(b_test, predicted))*100

accuracies['SVM'] = accuracy

print('Support Vector Machines accuracy: ', accuracy)

print('Confusion Matrix: ')

print(metrics.confusion_matrix(b_test, predicted))
plt.figure(figsize = (8,5))

plt.yticks(np.arange(0,100,10))

sns.barplot(x = list(accuracies.keys()), y = list(accuracies.values()))
y_cm_lr = logReg.predict(testData)

y_cm_nb = nb.predict(testData)

y_cm_knn = knn.predict(testData)

y_cm_svm = svm.predict(testData)
from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(b_test, y_cm_lr)

cm_knn = confusion_matrix(b_test, y_cm_knn)

cm_svm = confusion_matrix(b_test, y_cm_svm)

cm_nb = confusion_matrix(b_test, y_cm_nb)
plt.figure(figsize = (8,8))

plt.subplots_adjust(wspace = 0.4, hspace = 0.4)



plt.subplot(2,2,1)

plt.title("LR Confusion Matrix")

sns.heatmap(cm_lr,annot=True,fmt="d",cbar=False, annot_kws={"size": 12})



plt.subplot(2,2,2)

plt.title("KNN Confusion Matrix")

sns.heatmap(cm_knn, annot = True, fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(2,2,3)

plt.title("NB Confusion Matrix")

sns.heatmap(cm_nb, annot = True, fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(2,2,4)

plt.title("SVM Confusion Matrix")

sns.heatmap(cm_svm, annot = True, fmt = 'd', cbar = False, annot_kws = {"size": 12})