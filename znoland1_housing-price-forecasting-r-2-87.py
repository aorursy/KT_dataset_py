%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv('../input/kc_house_data.csv')

df.head()
# Review info

df.info()
plt.figure(figsize=(15,12))

mask = np.zeros_like(df.corr())

mask[np.triu_indices_from(mask)] = True

with sns.axes_style('white'):

    ax = sns.heatmap(df.corr(), mask=mask, vmax=.3, annot=True)
# Split label from X

y = df['price']

X = df.drop('price', axis=1)
# Convert yr_renovated to years since renovation

X['sold_year'] = X['date'].apply(lambda x: int(x[:4]))

X['yrs_since_renovated'] = (X['sold_year'] - X['yr_renovated'][X['yr_renovated'] != 0]).fillna(0)



# Create dummy features for zip code

zip_dummies = pd.get_dummies(X['zipcode'], prefix='zipcode')

X = pd.concat([X, zip_dummies], axis=1)





# Drop certain features now, revisit later to add

X = X.drop(['date', 'yr_renovated', 'sold_year', 'zipcode', 'id'], axis=1)





X.head()
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures



# Create Polynomial Features

poly = PolynomialFeatures(2)

X_poly = poly.fit_transform(X)



# Split train and test sets

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.4)



# Fit model

model = LinearRegression()

model.fit(X_train, y_train)



# Return MSE

print('Train set MSE: {}'.format(mean_squared_error(y_train, model.predict(X_train))))

print('Test set MSE: {}'.format(mean_squared_error(y_test, model.predict(X_test))))
from sklearn.metrics import r2_score



# Return R^2

print('Train Score: {:.2f}'.format(model.score(X_train, y_train)))

print('Test Score: {:.2f}'.format(model.score(X_test, y_test)))

#Kaggle kept timeing out when generating the Learning curve, so I commented it out



# Debug learning Curve

#from sklearn.model_selection import learning_curve

#

#train_sizes, train_scores, valid_scores = learning_curve(model, X_train, y_train, 

#                                                         train_sizes=np.linspace(.1, 1.0, 5), cv=5)
#plt.grid()

#plt.plot(train_sizes, train_scores, label='Training Score')

#plt.plot(train_sizes, valid_scores, label='Test Score')