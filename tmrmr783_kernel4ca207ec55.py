# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.|
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
sample = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
X = train
s = (X.dtypes == 'object')

obj_cols = list(s[s].index)

X = X.drop(columns=obj_cols)
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='mean')



y = X.SalePrice

y_old = X.SalePrice

X.drop(columns=['SalePrice', 'Id'], axis=1, inplace=True)



X_columns = X.columns

X = imputer.fit_transform(X)
X = np.log1p(X)

y = np.log1p(y)
X = pd.DataFrame(data=X[:,:], columns=X_columns)
X.describe()
X.head()
X_test = test[X.columns]

Id_y = train.Id

x_columns = X_test.columns
X_test = imputer.transform(X_test)

X_test = pd.DataFrame(data=X_test[:,:], columns=x_columns)

print(X_test.isnull().sum())
import torch
X_train = X

y_train = y
X_train = torch.FloatTensor(X_train.values)

y_train = torch.FloatTensor(y_train)
class Neuron(torch.nn.Module):

    def __init__(self,n_input, n_hidden_neurons):

        super(Neuron, self).__init__()

        self.bn1 = torch.nn.BatchNorm1d(n_input)

        self.fc1 = torch.nn.Linear(n_input, n_hidden_neurons)

        self.activ1 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)

        self.activ2 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(n_hidden_neurons, 1)

        self.activ3 = torch.nn.Sigmoid()

        

    def forward(self, x):

        x = self.bn1(x)

        x = self.fc1(x)

        x = self.activ1(x)

        x = self.fc2(x)

        x = self.activ2(x)

        x = self.fc3(x)

        x = self.activ3(x)

        return x

    

n_input = 36

n_hidden = 7

net = Neuron(n_input, n_hidden)
loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)
def loz(x, y):

    a = (x-y)**2

    return a.mean()
batch_size = 200



for epoch in range(200):

    order = np.random.permutation(1460)

    for start_index in range(0, 1460, batch_size):

        optimizer.zero_grad()

        

        batch_indexes = order[start_index:start_index+batch_size]

        

        x_batch = X_train[batch_indexes]

        y_batch = y_train[batch_indexes]

        

        preds = net.forward(x_batch) 

        

        loss_value = loss(preds, y_batch)

        loss_value.backward()

        

        optimizer.step()

    

    train_preds = net.forward(X_train)

    print(epoch, "-epoch number, loss =",loz(train_preds.detach().numpy(), y_train.detach().numpy()),"\n")
x_0 = torch.FloatTensor(x_test.values)
y_0 = net.forward(x_0)
y = y_0.detach().numpy()
y = np.ndarray.flatten(y)
predictions = pd.DataFrame({'Id': Id_y, 'SalePrice' : y})

predictions.head()
from sklearn import linear_model, metrics

import sklearn

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge
clf = LinearRegression

clf1 = Lasso

clf2 = Ridge
LinearRegression.get_params(clf)
Lasso.get_params(clf1).keys()
Ridge.get_params(clf2).keys()
parameters_grid = {

    'alpha': np.linspace(0.0001, 0.1, num=10),

    'fit_intercept': [True, False],

    'normalize': [True, False],

    'random_state': range(1,26)

}
cv = sklearn.model_selection.StratifiedShuffleSplit(n_splits=7, test_size=0.2, random_state=0)
grid_cv = sklearn.model_selection.GridSearchCV(clf2, parameters_grid, scoring='accuracy', cv=cv)
grid_cv.fit(X,y)
X_train = X
kf = KFold(n_splits=5, shuffle=True)
def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
alphas = np.linspace(0.05, 10)

betas = np.linspace(0.05, 10)
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_lasso = [rmse_cv(Ridge(alpha = beta)).mean() for beta in betas]
import matplotlib.pyplot as plt
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_lasso = pd.Series(cv_lasso, index = betas)

cv_lasso.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
X_test = test[X.columns]

Id_y = train.Id

x_columns = X_test.columns

X_test = imputer.transform(X_test)
X_test = np.log1p(X_test)
clf = Lasso(alpha = 0.001)
clf.fit(X_train,y)
clf.predict(X_test)
answer = np.expm1(clf.predict(X_test))
Id = [i+1461 for i in range(1459)]
len(Id), answer.shape
answer.tolist()
predictions = pd.DataFrame({'Id': Id, 'SalePrice' : answer.tolist()})

predictions.head()
predictions.tail()
predictions.to_csv('predictions.csv', index = False)