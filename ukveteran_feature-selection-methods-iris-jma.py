import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

df.head()
df = df.drop(['species'],axis=1) # Dropping Multiple Columns

df.head()
X = df.iloc[:,:-1]

y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state= 10)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)

y_predict = lr.predict(X_test)
from sklearn.metrics import mean_squared_error

import math

rsme = math.sqrt(mean_squared_error(y_test,y_predict))

rsme
from sklearn.feature_selection import f_regression as fr

result = fr(X,y)

f_score = result[0]

p_values = result[1]



# Getting column names 

columns = list(X.columns)

print(" ")

print(" ")

print(" ")



print("     Features                     ","F-Score  ","P-Values")

print("     ------------                   --------  ---------")



for i in range(0,len(columns)):

    f1 = "%4.2f" % f_score[i]

    p1 = "%2.6f" % p_values[i]

    print("    ",columns[i].ljust(25),f1.rjust(12),"",p1.rjust(8))
X_train_n = X_train[['sepal_length','sepal_width','petal_length']]

X_test_n = X_test[['sepal_length','sepal_width','petal_length']]
from sklearn.linear_model import LinearRegression

lr1 = LinearRegression()

lr1.fit(X_train_n,y_train)

y_predict1 = lr1.predict(X_test_n)
from sklearn.metrics import mean_squared_error

import math

rsme1 = math.sqrt(mean_squared_error(y_test,y_predict1))

rsme1
from sklearn.feature_selection import f_regression as fr

from sklearn.feature_selection import GenericUnivariateSelect

selectorG1 = GenericUnivariateSelect(score_func=fr,mode='k_best',param=3)

X_g1 = selectorG1.fit_transform(X,y)
# get the column names 

cols_g1 = selectorG1.get_support(indices = True)

selectedCols_g1 = X.columns[cols_g1].tolist()

print(selectedCols_g1)
from sklearn.feature_selection import f_regression as fr

from sklearn.feature_selection import GenericUnivariateSelect

selectorG2 = GenericUnivariateSelect(score_func=fr,mode='percentile',param=20)

X_g2 = selectorG2.fit_transform(X,y)
# get the column names 

cols_g2 = selectorG2.get_support(indices = True)

selectedCols_g2 = X.columns[cols_g2].tolist()

print(selectedCols_g2)
from sklearn.ensemble import RandomForestRegressor

np.random.seed()

forest = RandomForestRegressor(n_estimators=1000)

fit = forest.fit(X_train,y_train)

accuracy = fit.score(X_test,y_test)

predict = fit.predict(X_test)

#cmatrix = confusion_matrix(y_test,predict)



#-------------------------------------------------------------------------------------------------#

# Perform k Fold cross- validation 



print('Accuracy of Random Forest: %s'% "{0:.2%}".format(accuracy))
# Feature importance 

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]
import matplotlib.pyplot as plt

import seaborn as sns

print("Feature ranking")

for f in range(X.shape[1]):

    print("Feature %s (%f)" % (list(X)[f],importances[indices[f]]))



feat_imp = pd.DataFrame({'Feature':list(X),

                        'Gini importance':importances[indices]})

plt.rcParams['figure.figsize']=(12,12)

sns.set_style('whitegrid')

ax = sns.barplot(x='Gini importance',y='Feature',data=feat_imp)

ax.set(xlabel='Gini Importance')

pass
from sklearn.linear_model import Lasso 

from sklearn.feature_selection import SelectFromModel

sel_ = SelectFromModel(Lasso(alpha=0.005,random_state=0))

sel_.fit(X_train,y_train)
sel_.get_support()
selected_feat=X_train.columns[(sel_.get_support())]



print('total features: {}'.format((X_train.shape[1])))

print('Selected features: {}'.format(len(selected_feat)))

print('Features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_==0)))
selected_feat
selected_feat = X_train.columns[(sel_.estimator_.coef_!=0).ravel().tolist()]

selected_feat