# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/column_2C_weka.csv")

data.head()
data.corr()
f, axis = plt.subplots(figsize=(12,12))

sns.heatmap(data.corr(), annot=True, linewidths=0.4, fmt='.2f', ax = axis)

plt.show()
sns.countplot(x="class", data=data)

data.loc[:, 'class'].value_counts()
sns.pairplot(data,hue="class",palette="Set2", diag_kind = 'kde')

plt.show()
from sklearn.linear_model import LinearRegression



data1=data[data['class'] == "Abnormal"]



linear_Reg = LinearRegression()

x = np.array(data1.loc[:, 'pelvic_incidence']).reshape(-1,1)

y = np.array(data1.loc[:, 'sacral_slope']).reshape(-1,1)



linear_Reg.fit(x, y)

y_head = linear_Reg.predict(x)
plt.figure(figsize=(15, 5))

plt.scatter(x, y, color="green")

plt.plot(x, y_head, color="black")

plt.xlabel("pelvic_incidence")

plt.ylabel("sacrel_scope")

plt.show()
linear_Reg = LinearRegression()

x = np.array(data1.loc[:, 'pelvic_tilt numeric']).reshape(-1,1)

y = np.array(data1.loc[:, 'lumbar_lordosis_angle']).reshape(-1,1)



linear_Reg.fit(x, y)

y_head = linear_Reg.predict(x)
plt.figure(figsize=(15, 5))

plt.scatter(x, y, color="green")

plt.plot(x, y_head, color="black")

plt.xlabel("pelvic_tilt numeric")

plt.ylabel("lumbar_lordosis_angle")

plt.show()
data.columns
#split dataset in features and target variable

feature_cols = ['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle',

       'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']



X = data[feature_cols] # Features

y = data['class'] # Target variable
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# import the class

from sklearn.linear_model import LogisticRegression



# instantiate the model (using the default parameters)

logreg = LogisticRegression()



# fit the model with data

logreg.fit(X_train,y_train)



# predict

y_pred = logreg.predict(X_test)
# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
class_names=["Abnormal", "Normal"] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)



# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred, labels=['Normal', 'Abnormal'], average=None))

print("Recall:",metrics.recall_score(y_test, y_pred, labels=['Normal', 'Abnormal'], average=None))
y_pred_proba = logreg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba, pos_label="Abnormal")

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
y_pred_proba = logreg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba, pos_label="Normal")

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()