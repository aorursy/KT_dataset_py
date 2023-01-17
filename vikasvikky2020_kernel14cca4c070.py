# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test=pd.read_csv('/kaggle/input//mobile-price-range-prediction-is2020-v2/test_data.csv')
train=pd.read_csv('/kaggle/input//mobile-price-range-prediction-is2020-v2/train_data.csv')
sample_submission=pd.read_csv('/kaggle/input//mobile-price-range-prediction-is2020-v2/sample_submission.csv')
test.head()
train.head()
train.shape
train.describe()
y=train['price_range']
x=train.drop('price_range',axis=1)

y.unique()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.01, random_state = 101, stratify = y)

#x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 101, stratify = y)
print(x_train.shape)
print(x_valid.shape)
fig = plt.subplots (figsize = (12, 12))
sns.heatmap(train.corr (), square = True, cbar = True, annot = True, cmap="GnBu", annot_kws = {'size': 8})
plt.title('Correlations between Attributes')
plt.show ()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(multi_class = 'multinomial', solver = 'sag',  max_iter = 10000)
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_valid)
from sklearn import metrics
from sklearn.metrics import accuracy_score
confusion_matrix = metrics.confusion_matrix(y_valid, y_pred_lr)
confusion_matrix
acc_lr = metrics.accuracy_score(y_valid, y_pred_lr)
acc_lr
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
import math
dt = DecisionTreeClassifier(random_state=101)
dt_model = dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_valid)

dt_model

print(metrics.confusion_matrix(y_valid, y_pred_dt))
print(metrics.classification_report(y_valid, y_pred_dt))
acc_dt = metrics.accuracy_score(y_valid, y_pred_dt)
acc_dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state=101, criterion = 'entropy', oob_score = True) 
model_rf = rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_valid)
print(metrics.confusion_matrix(y_valid, y_pred_rf))
pd.crosstab(y_valid, y_pred_rf, rownames=['Actual Class'], colnames=['Predicted Class'])
acc_rf = metrics.accuracy_score(y_valid, y_pred_rf)
acc_rf
model_knn = KNeighborsClassifier(n_neighbors=3)  
model_knn.fit(x_train, y_train)
y_pred_knn = model_knn.predict(x_valid)
print(metrics.confusion_matrix(y_valid, y_pred_knn))
print(accuracy_score(y_valid, y_pred_knn))
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors':np.arange(1,30)}
knn = KNeighborsClassifier()

model = GridSearchCV(knn, parameters, cv=5)
model.fit(x_train, y_train)
model.best_params_
model_knn = KNeighborsClassifier(n_neighbors=9)  
model_knn.fit(x_train, y_train)
y_pred_knn = model_knn.predict(x_valid)
print(metrics.confusion_matrix(y_valid, y_pred_knn))
acc_knn = accuracy_score(y_valid, y_pred_knn)
acc_knn
models = ['logistic regression', 'decision tree', 'random forest', 'knn']
acc_scores = [0.73, 0.83, 0.90, 0.95]

plt.bar(models, acc_scores, color=['lightblue', 'pink', 'lightgrey', 'cyan'])
plt.ylabel("accuracy scores")
plt.title("Which model is the most accurate?")
plt.show()
test.head()
predicted_price_range = model_knn.predict(test)
predicted_price_range 
train.head()
data={'id':sample_submission['id'],
     'price_range':predicted_price_range}
result=pd.DataFrame(data)
result.to_csv("/kaggle/working/result_12.csv",index=False)
output=pd.read_csv('/kaggle/working/result_12.csv')