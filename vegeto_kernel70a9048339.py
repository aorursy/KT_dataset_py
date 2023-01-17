# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings('ignore')
pima = pd.read_csv("../input/diabetes.csv")
pima.head()
pima.shape
pima.describe()
pima.hist(figsize=(10,8))
pima.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))
column_x = pima.columns[0:len(pima.columns) - 1]
column_x
corr = pima[pima.columns].corr()
sns.heatmap(corr, annot = True)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = pima.iloc[:,0:8]
Y = pima.iloc[:,8]
select_top_4 = SelectKBest(score_func=chi2, k = 4)
fit = select_top_4.fit(X,Y)
features = fit.transform(X)
features[0:5]
pima.head()
X_features = pd.DataFrame(data = features, columns = ["Glucose","Insulin","BMI","Age"])
X_features.head()

Y = pima.iloc[:,8]
Y.head()
from sklearn.preprocessing import StandardScaler
rescaledX = StandardScaler().fit_transform(X_features)
X = pd.DataFrame(data = rescaledX, columns= X_features.columns)
X.head()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, random_state = 22, test_size = 0.2)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("DT",DecisionTreeClassifier()))
models.append(("SVM",SVC()))
results = []
names = []
for name,model in models:
    kfold = KFold(n_splits=10, random_state=22)
    cv_result = cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "accuracy")
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean())
ax = sns.boxplot(data=results)
ax.set_xticklabels(names)
lr = LogisticRegression()
lr.fit(X_train,Y_train)
predictions = lr.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(accuracy_score(Y_test,predictions))
svm = SVC()
svm.fit(X_train,Y_train)
predictions = svm.predict(X_test)
print(accuracy_score(Y_test,predictions))
print(classification_report(Y_test,predictions))
conf = confusion_matrix(Y_test,predictions)
label = ["0","1"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)