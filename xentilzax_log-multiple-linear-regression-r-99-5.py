import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



import copy
data = pd.read_csv('../input/fish-market/Fish.csv')
data.head()
data.describe()
data.drop(data[data["Weight"] == 0].index, inplace = True)

data.describe()
sns.countplot(x='Species',data=data)
sns.heatmap(data.corr(), annot=True, cmap='YlGnBu');
data.drop(columns=['Length2', 'Length3'], inplace = True)

data.head()
g = sns.FacetGrid(data, col="Species")

g.map(plt.scatter, "Length1", "Weight", alpha=.7)

g.add_legend();
X = pd.get_dummies(data)

X.drop(columns=['Weight','Length1','Height','Width'], inplace=True)



i = 0



columns_L = X.columns[i:(i+7)]

columns_L = [s.replace('Species', 'L') for s in columns_L]

X.rename(columns= {X.columns[i+0]:columns_L[0], 

                   X.columns[i+1]:columns_L[1], 

                   X.columns[i+2]:columns_L[2],

                   X.columns[i+3]:columns_L[3],

                   X.columns[i+4]:columns_L[4],

                   X.columns[i+5]:columns_L[5],

                   X.columns[i+6]:columns_L[6],

                   }, inplace=True)



columns_H = [s.replace('L', 'H') for s in columns_L]

columns_W = [s.replace('L', 'W') for s in columns_L]



for k in range(7):

    X[columns_H[k]] = X[columns_L[k]]

    X[columns_W[k]] = X[columns_L[k]]

    

    X[columns_L[k]] *= data['Length1']

    X[columns_H[k]] *= data['Height']

    X[columns_W[k]] *= data['Width']

X.head()
X[X > 0] = np.log(X[X > 0])

y = data['Weight']

y = np.log(y)

X.head()
model = LinearRegression()

model.fit(X,y);
y_pred = model.predict(X)

y_exp = np.exp(y)

y_pred = np.exp(y_pred)

print("R2: ", r2_score(y_exp, y_pred))

error = y_exp - y_pred



plt.scatter(y_pred, error);

plt.xlabel('weight')

plt.ylabel('error')

plt.style.use('_classic_test_patch')
error = error/y_pred
sns.distplot(error);

plt.title('Residual Graph');
th = 0.07

q_max = error.quantile(1 - th)

q_min = error.quantile(th)

print("max", q_max)

print("min", q_min)

idx = error[(error > q_max) | (error < q_min)].index

sns.distplot(error.drop(idx));
X_cleared = X.drop(idx)

y_cleared = y.drop(idx)
model = LinearRegression()

model.fit(X_cleared, y_cleared)



y_pred2 = model.predict(X_cleared)

y_exp2 = np.exp(y_cleared)

y_pred2 = np.exp(y_pred2)



r2_score(y_exp2, y_pred2)
#R2 Score for not cleared data

y_pred2 = model.predict(X)

y_exp2 = np.exp(y)

y_pred2 = np.exp(y_pred2)



r2_score(y_exp2, y_pred2)
y_dif2 = y_exp2 - y_pred2

plt.scatter(y_pred2, y_dif2);

plt.xlabel('weight')

plt.ylabel('error');