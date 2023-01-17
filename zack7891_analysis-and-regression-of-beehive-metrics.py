#This Kernal is still a work in progress



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

flow = pd.read_csv("../input/flow_2017.csv")

humid = pd.read_csv("../input/humidity_2017.csv")

temp = pd.read_csv("../input/temperature_2017.csv")

weight = pd.read_csv("../input/weight_2017.csv")



print(flow.head())

print(humid.head())

print(temp.head())

print(weight.head())



timestamp = flow['timestamp']

flow_1 = flow['flow']

humid_1 = humid['humidity']

temp_1 = temp['temperature']

weight_1 = weight['weight']



print(flow_1.count())

print(humid_1.count())

print(temp_1.count())

print(weight_1.count())



finaldf = pd.concat([flow_1, humid_1, temp_1, weight_1], axis=1, join='inner').sort_index()

#Correlation Map



import seaborn as sns

corr_mat = finaldf.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
import statsmodels.api as sm

from statsmodels.formula.api import ols



model_lm = ols('flow ~ humidity + temperature + weight', 

               data = finaldf).fit()

aov_table = sm.stats.anova_lm(model_lm, typ=2)

print(aov_table)



esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])

print('\n',esq_sm)



X = finaldf[["humidity","temperature","weight"]].head(8737)

y = flow_1.head(8737)

print(X.describe())

print('\n', y.describe())





model = sm.OLS(y,X).fit()



pred = model.predict(X)



print(model.summary())

print(model.params)
import sklearn

from sklearn.feature_selection import chi2

from sklearn.linear_model import LogisticRegression



#Logistic Regression of Flow 

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)

print(clf.predict_proba(X))

print(clf.score(X, y))
#Bayesian

clf2 = sklearn.linear_model.BayesianRidge()

print(clf2.fit(X,y))

print(clf2.predict(X))
print(sklearn.feature_selection.f_regression(X, y))
import seaborn as sns

X = finaldf[["humidity","temperature"]].head(8737)

y = weight_1.head(8737)

print(X.describe())

print('\n', y.describe())





print("\n Regression of Weight")

model = sm.OLS(y,X).fit()



pred = model.predict(X)



print(model.summary())

print(model.params)



print("\n")



print("ANOVA")

model_lm = ols('weight ~ humidity + temperature', 

               data = finaldf).fit()



aov_table = sm.stats.anova_lm(model_lm, typ=2)

print(aov_table)



esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])

print('\n',esq_sm)


