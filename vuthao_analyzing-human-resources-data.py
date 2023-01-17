import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_auc_score

%matplotlib inline
hr = pd.read_csv('../input/HR_comma_sep.csv')
print(hr.shape)

hr.head()
fig, ((a,b),(c,d)) = plt.subplots(2,2, figsize= (15,20))

plt.xticks(rotation=70)

sns.countplot(hr['Work_accident'],hue=hr['left'],ax=a)



sns.countplot(hr['sales'],hue=hr['left'],ax=b)



sns.countplot(hr['salary'],hue=hr['left'],ax=c)



sns.countplot(hr['promotion_last_5years'],hue=hr['left'],ax=d)

hr['salary'].replace({'low':1,'medium':5,'high':10},inplace=True)

hr = pd.concat((hr, pd.get_dummies(hr["sales"])), axis = 1, join_axes= [hr.index])

hr.drop("sales", axis = 1, inplace = True)

corr= hr[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident",

         "promotion_last_5years","salary"]].corr()

sns.heatmap(corr,annot=True)
hr.head()
target = list(hr["left"])

hr.drop("left", axis = 1, inplace = True)
hr.shape
from sklearn.cross_validation import train_test_split

train_vector, valid_vector, train_target, valid_target = train_test_split(hr, target, test_size = 0.2)
print(train_vector.shape)

print(valid_vector.shape)

print (len(train_target))

print(len(valid_target))
import sklearn

from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators = 100)

rf.fit(train_vector, train_target)

pred = rf.predict(train_vector)

print(sklearn.metrics.f1_score(train_target, pred, average='binary'))

pred = rf.predict(valid_vector)

print(sklearn.metrics.f1_score(valid_target, pred, average='binary'))
num_est = [50,100,150,200,300,400]

acc = []

f1 = []

roc = []

for i in num_est:

    rf= RandomForestClassifier(n_estimators = i)

    rf.fit(train_vector, train_target)

    

    pred = rf.predict(valid_vector)

    acc.append(np.mean(pred==valid_target))

    f1.append(sklearn.metrics.f1_score(valid_target, pred, average='binary'))

    roc.append(roc_auc_score(valid_target, pred))

    
import matplotlib.pylab as plt

plt.plot(num_est, f1, color = "blue", label = "F1 Score")

plt.plot(num_est, acc, color = "red", label = "accuracy")

plt.plot(num_est, roc, color = "green", label = 'ROC')

plt.xlabel("Number of Estimators")

plt.ylabel("Score")

plt.legend(loc = "best")

plt.xlim((0,450))

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
num_est = [50,100,150,200,300,400]

acc = []

f1 = []

roc = []

for i in num_est:

    dt= DecisionTreeClassifier(max_depth=3,min_samples_leaf=int(0.05*len(train_vector)),random_state=19)

    boosted_dt=AdaBoostClassifier(dt,algorithm='SAMME',n_estimators=800,learning_rate=0.5)

    boosted_dt.fit(train_vector, train_target)

    

    pred = boosted_dt.predict(valid_vector)

    acc.append(np.mean(pred==valid_target))

    f1.append(sklearn.metrics.f1_score(valid_target, pred, average='binary'))

    roc.append(roc_auc_score(valid_target, pred))

    
plt.plot(num_est, f1, color = "blue", label = "F1 Score")

plt.plot(num_est, acc, color = "red", label = "accuracy")

plt.plot(num_est, roc, color = "green", label = 'ROC')

plt.xlabel("Number of Estimators")

plt.ylabel("Score")

plt.legend(loc = "best")

plt.xlim((0,450))
