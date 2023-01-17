

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')



import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
#df.head()
#df.describe()
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x=df.pelvic_incidence.values.reshape(-1,1)

y=df.sacral_slope.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head=linear_reg.predict(x)

# Regression line and scatter plot

plt.figure(figsize=[7,7])

plt.scatter(x,y)

plt.xlabel("pelvic incidence")

plt.ylabel("sacral slope")

plt.plot(x,y_head,color="brown")

plt.show()
# Prediction

b0=linear_reg.predict([[0]])

b1=linear_reg.coef_

print("b0: ",b0)

print("b1: ",b1)



print("Predict: ",b1*60+b0)
# r square score

from sklearn.metrics import r2_score

print("R square score : ",r2_score(y,y_head))

from sklearn.linear_model import LinearRegression



x=df.iloc[:,[1,2,3,4]].values

y=df.pelvic_incidence.values.reshape(-1,1)



multiple_reg=LinearRegression()

multiple_reg.fit(x,y)

print("b0: ",multiple_reg.intercept_)

print("b1,b2,b3,b4,: ", multiple_reg.coef_)
y_head=multiple_reg.predict(x)

from sklearn.metrics import r2_score

print("r square score :",multiple_reg.score(x,y))
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(x)
from sklearn.linear_model import LinearRegression

plt.figure(figsize=[10,10])

poly_reg=LinearRegression()

poly_reg.fit(x_polynomial,y)

y_head=poly_reg.predict(x_polynomial)

plt.plot(x,y_head,color="brown")

plt.show()


from sklearn.metrics import r2_score

print("r square score : ",r2_score(y,y_head))
from sklearn.tree import DecisionTreeRegressor

import numpy as np

x=df.pelvic_incidence.values.reshape(-1,1)

y=df.sacral_slope.values.reshape(-1,1)

tree=DecisionTreeRegressor()

tree.fit(x,y)



x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=tree.predict(x_)

plt.plot(x_,y_head,color="brown")

plt.show()
x=df.pelvic_incidence.values.reshape(-1,1)

y=df.sacral_slope.values.reshape(-1,1)

plt.figure(figsize=[20,20])

plt.scatter(x,y)



from sklearn.ensemble import RandomForestRegressor



random_reg=RandomForestRegressor(n_estimators=100,random_state=42)



random_reg.fit(x,y)



x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=random_reg.predict(x_)

plt.plot(x_,y_head,color="brown")

plt.show()