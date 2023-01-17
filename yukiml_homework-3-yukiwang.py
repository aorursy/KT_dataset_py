!pip install regressors
import numpy as np 

import pandas as pd 

from regressors import stats

from sklearn import linear_model as lm

import statsmodels.formula.api as sm

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression 

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



import os

print(os.listdir("../input"))
q1=pd.read_csv("../input/smarket.csv")

q1.head(10)
print("Summary for smarket\n",q1.describe())
a = q1[["Year", "Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume", "Today"]]

print(a.isnull().values.any())

print(a.isnull().sum())

fig = plt.figure(8, figsize=(20, 12))

cols = ["Year","Lag1","Lag2","Lag3","Lag4","Lag5","Volume","Today"]

for i in range(0,len(cols)):

    ax = fig.add_subplot(241+i)

    ax.boxplot(a[cols[i]])

plt.show()
#Add a new column DefaultYes which is 1 for Yes and 0 for No

q1['DirectionUp'] = q1['Direction'].map({'Up': 1, 'Down': 0})

q1.head()
inputDF = q1[["Year","Lag1","Lag2","Lag3","Lag4","Lag5","Volume","Today"]]

outputDF = q1[["DirectionUp"]].values.ravel()



logisticRegr = LogisticRegression(solver='lbfgs')

x_train, x_test, y_train, y_test = train_test_split(inputDF, outputDF, test_size=0, random_state=0)

logisticRegr.fit(inputDF, outputDF)



print(logisticRegr.intercept_)

print(logisticRegr.coef_)
# Add prediction

y_pred = logisticRegr.predict(inputDF)

q1['Prediction'] = y_pred

q1.head()
q1['Correct'] = q1['DirectionUp'] == q1['Prediction']

q1.head()

oneTrue = (q1["Correct"] == True) & (q1["Prediction"] == 1)

print("Up and Correct: {}".format(dict(oneTrue.value_counts())[True]))

oneFalse = (q1["Correct"] == False) & (q1["Prediction"] == 1)

print("Up and Wrong: {}".format(dict(oneFalse.value_counts())[True]))

zeroTrue = (q1["Correct"] == True) & (q1["Prediction"] == 0)

print("Down and Correct: {}".format(dict(zeroTrue.value_counts())[True]))



zeroFalse = (q1["Correct"] == False) & (q1["Prediction"] == 0)

val_count = dict(zeroFalse.value_counts())

print("Down and Wrong: {}".format(0 if True not in val_count else val_count[True]))
q1['Correct'].value_counts()
#Logistic Regression with Training/Test Separation

q2=pd.read_csv("../input/smarket.csv")

q2.head(10)
testset = q2[q2.Year == 2005]

trainset = q2[q2.Year != 2005]

trainset['DirectionUp'] = trainset['Direction'].map({'Up': 1, 'Down': 0})
inputDF = trainset[["Year","Lag1","Lag2","Lag3","Lag4","Lag5","Volume","Today"]]

outputDF = trainset[["DirectionUp"]].values.ravel()



logisticRegr = LogisticRegression(solver='lbfgs')

x_train, x_test, y_train, y_test = train_test_split(inputDF, outputDF, test_size=0, random_state=0)

logisticRegr.fit(inputDF, outputDF)



print(logisticRegr.intercept_)

print(logisticRegr.coef_)
# Add prediction

newinput = testset[["Year","Lag1","Lag2","Lag3","Lag4","Lag5","Volume","Today"]]

y_pred = logisticRegr.predict(newinput)

testset['Prediction'] = y_pred

testset.head()
uptrue = (testset["Prediction"] == 1) & (testset["Direction"] == "Up")

print("Up and Correct: {}".format(dict(uptrue.value_counts())[True]))

upfalse = (testset["Prediction"] == 1) & (testset["Direction"] == "Down")

val_count = dict(upfalse.value_counts())

print("Up and Wrong: {}".format(0 if True not in val_count else val_count[True]))

downtrue = (testset["Prediction"] == 0) & (testset["Direction"] == "Down")

print("Down and Correct: {}".format(dict(downtrue.value_counts())[True]))

downfalse = (testset["Prediction"] == 0) & (testset["Direction"] == "Up")

print("Down and Wrong: {}".format(dict(downfalse.value_counts())[True]))
