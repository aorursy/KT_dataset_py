#import the necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline
#reading the files

train = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

#viewing the data

train.head()
#taking a look at all the columns in the data

train.columns.values
sns.pairplot(train)
y = train['Chance of Admit '] #target variable

z = train['Research'] #extract it for later preprocessing as this only has boolean values

X = train.drop(columns=['Serial No.','Chance of Admit ','Research']) #drop the columns

print(X.head())

columns = X.columns.values

print(columns)
X.corr()
#Preprocessing data to get better results

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_new = scaler.fit_transform(X)

X_new 
X_new = pd.DataFrame(data=X_new,columns=['GRE_Score','TOEFL_Score','University_Rating','SOP','LOR','CGPA'])

X_new.head()
X_new['Research'] = z

X_new['Chances_of_admit'] = y
#importing more libraries to check the assumptions of Linear Regression

import statsmodels.formula.api as smf

import statsmodels.stats.api as sms

from scipy import stats

from statsmodels.compat import lzip
model = smf.ols("Chances_of_admit~GRE_Score+TOEFL_Score+SOP+LOR+CGPA+Research+University_Rating", data= X_new).fit()

    

model.summary()
pred_val = model.fittedvalues.copy()

true_val = y.values.copy()

residual = true_val - pred_val
fig, ax = plt.subplots(figsize=(6,2.5))

_ = ax.scatter(residual, pred_val)
stats.probplot(model.resid, plot= plt)

plt.title("Model Residuals Probability Plot")
#another measure of the degree of normal distribution of the residuals

stats.kstest(model.resid, 'norm')


name = ['Lagrange multiplier statistic', 'p-value', 

        'f-value', 'f p-value']

test = sms.het_breuschpagan(model.resid, model.model.exog)

lzip(name, test)
from sklearn.linear_model import LinearRegression

X_new = X_new.drop(columns=['Chances_of_admit'])

X_new['Research'] = z

reg = LinearRegression().fit(X_new, y)

reg.score(X_new, y)
#dropping GRE Score

X_new_drop= X_new.drop(columns=['GRE_Score'])

reg = LinearRegression().fit(X_new_drop, y)

reg.score(X_new_drop, y)
#dropping TOEFL Score

X_new_drop= X_new.drop(columns=['TOEFL_Score'])

reg = LinearRegression().fit(X_new_drop, y)

reg.score(X_new_drop, y)
#dropping University Rating

X_new_drop= X_new.drop(columns=['University_Rating'])

reg = LinearRegression().fit(X_new_drop, y)

reg.score(X_new_drop, y)
#dropping CGPA

X_new_drop= X_new.drop(columns=['CGPA'])

reg = LinearRegression().fit(X_new_drop, y)

reg.score(X_new_drop, y)
#dropping Research

X_new_drop= X_new.drop(columns=['Research'])

reg = LinearRegression().fit(X_new_drop, y)

reg.score(X_new_drop, y)
#dropping University Rating

X_new_drop= X_new.drop(columns=['University_Rating'])

reg = LinearRegression().fit(X_new_drop, y)

reg.score(X_new_drop, y)
#dropping LOR

X_new_drop= X_new.drop(columns=['LOR'])

reg = LinearRegression().fit(X_new_drop, y)

reg.score(X_new_drop, y)
#dropping SOP

X_new_drop= X_new.drop(columns=['SOP'])

reg = LinearRegression().fit(X_new_drop, y)

reg.score(X_new_drop, y)
test = pd.read_csv('../input/Admission_Predict.csv')

test.head()
test.columns.values
y_act = test['Chance of Admit ']

X_test = test.drop(columns=["Serial No.",'Chance of Admit '])
scaler = StandardScaler()

z = X_test['Research']

X_test = X_test.drop(columns=["Research"])

X_test_new = scaler.fit_transform(X_test)

X_test_new
X_test_new = pd.DataFrame(data=X_test_new,columns=['GRE_Score','TOEFL_Score','University_Rating','SOP','LOR','CGPA'])

X_test_new.head()
X_test_new['Research'] = z

X_test_new.head()
reg = LinearRegression().fit(X_new, y)

reg.score(X_new, y)
y_pred = reg.predict(X_test_new)
#evaluating the metrics

from sklearn import metrics

print(metrics.mean_absolute_error(y_act,y_pred))

print(metrics.mean_squared_error(y_act,y_pred))

print(np.sqrt(metrics.mean_squared_error(y_act,y_pred)))