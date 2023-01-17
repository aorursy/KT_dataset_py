import pandas as pd

import numpy as np

import os

print(os.listdir("../input"))



df = pd.read_csv('../input/jobs-on-naukricom/naukri_com-job_sample.csv')

df.head()
df.shape
df.describe()
df.info()
df['company'] = df['company'].astype(str)

df['education'] = df['education'].astype(str)

df['experience'] = df['experience'].astype(str)

df['industry'] = df['industry'].astype(str)

df['jobdescription'] = df['jobdescription'].astype(str)

df['jobid'] = df['jobid'].astype(str)

df['joblocation_address'] = df['joblocation_address'].astype(str)

df['jobtitle'] = df['jobtitle'].astype(str)

df['numberofpositions'] = df['numberofpositions'].astype(str)

df['payrate'] = df['payrate'].astype(str)

df['postdate'] = df['postdate'].astype(str)

df['site_name'] = df['site_name'].astype(str)

df['skills'] = df['skills'].astype(str)

df['uniq_id'] = df['uniq_id'].astype(str)
df.info()
from sklearn.preprocessing import LabelEncoder



# convert str values to int using the scikit-learn encoder



st = df.apply(LabelEncoder().fit_transform)



st.head()

df = st
X = df.iloc[:,:-1].values

y = df.iloc[:,1].values
X
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error , r2_score

mse = mean_squared_error(y_test,y_pred)

r_squared = r2_score(y_test,y_pred)

print(mse)

print(r_squared)
import statsmodels

import statsmodels.api as smf

regressorlm = X_train 

regressorlm = smf.add_constant(regressorlm)

# create a fitting model in one time

regressorlm=smf.OLS(y_train,regressorlm).fit()

regressorlm.summary()
regressor.intercept_
regressor.coef_