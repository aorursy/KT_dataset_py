import numpy as np

import pandas as pd

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.ensemble import VotingRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline
data = pd.read_csv('../input/startup-logistic-regression/50_Startups.csv')

print('input: R&D, Administration, Marketing, State')

print('output: Profit')

data.head(11)
print('basic statistical details:')

data.describe()
print('relationships between the data:')

sns.pairplot(data)

sns.pairplot(data, hue = 'State')
sns.heatmap(data.corr(), annot = True)
data['State'] = LabelEncoder().fit_transform(data['State'])

data.head(11)
Y = data['Profit']

X = data.drop('Profit', axis = 1)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.5, random_state = 21)
lasso_regression = Lasso(alpha = 1, normalize = True, max_iter = 3000)



lasso_regression.fit(trainX, trainY)

lasso_result = lasso_regression.predict(testX)



train_err = 1 - lasso_regression.score(trainX, trainY)

test_err = 1 - lasso_regression.score(testX, testY)



print('train error =', train_err, '\ntest error =', test_err, '\ndifference =', abs(train_err - test_err))
ridge_regression = Ridge(alpha = 1, normalize = True, solver = 'auto')



ridge_regression.fit(trainX, trainY)

ridge_result = ridge_regression.predict(testX)



train_err = 1 - ridge_regression.score(trainX, trainY)

test_err = 1 - ridge_regression.score(testX, testY)



print('train error =', train_err, '\ntest error =', test_err, '\ndifference =', abs(train_err - test_err))
elnet_regression = ElasticNet(alpha = 1, normalize = True, max_iter = 3000)



elnet_regression.fit(trainX, trainY)

elnet_result = elnet_regression.predict(testX)



train_err = 1 - elnet_regression.score(trainX, trainY)

test_err = 1 - elnet_regression.score(testX, testY)



print('train error =', train_err, '\ntest error =', test_err, '\ndifference =', abs(train_err - test_err))
ensemble = VotingRegressor([('Lasso', lasso_regression), ('Ridge', ridge_regression), ('ElNet', elnet_regression)], weights = [0.5, 0.5, 0.5])



ensemble.fit(trainX, trainY)

ensemble_result = ensemble.predict(testX)



train_err = 1 - ensemble.score(trainX, trainY)

test_err = 1 - ensemble.score(testX, testY)



print('train error =', train_err, '\ntest error =', test_err, '\ndifference =', abs(train_err - test_err))
import matplotlib.pyplot as plt



plt.figure()

plt.plot(lasso_result, 'gd', label = 'Lasso')

plt.plot(ridge_result, 'b^', label = 'Ridge')

plt.plot(elnet_result, 'ys', label = 'ElNet')

plt.plot(ensemble_result, 'r*', ms = 10, label = 'VotingRegressor')



plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)

plt.ylabel('predicted')

plt.xlabel('training samples')

plt.legend(loc = "best")

plt.title('Regressor predictions and their average')



plt.show()