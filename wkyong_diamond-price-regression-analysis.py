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
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor

import sklearn

from xgboost.sklearn import XGBRegressor

import xgboost as xgb

%matplotlib inline
plt.style.use("seaborn-dark")
os.chdir("../input/diamonds")
df_diamonds = pd.read_csv("diamonds.csv")
df_diamonds.head()
clarity_values = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
clarity_mapping = list(range(1, len(clarity_values) + 1))

cut_values = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
cut_mapping = list(range(1, len(cut_values) + 1))

color_values = np.sort(df_diamonds['color'].unique())
color_mapping = list(range(len(color_values), 0, -1))
proc_df_diamonds = df_diamonds.copy()
proc_df_diamonds['clarity'].replace(clarity_values, clarity_mapping, inplace = True)
proc_df_diamonds['cut'].replace(cut_values, cut_mapping, inplace = True)
proc_df_diamonds['color'].replace(color_values, color_mapping, inplace = True)
proc_df_diamonds = proc_df_diamonds.drop(proc_df_diamonds.columns[0], axis=1)
proc_df_diamonds.info()
proc_df_diamonds.head()
plt.figure(figsize = (15,15))
sns.heatmap(proc_df_diamonds.corr(), annot = True)
fig, ax = plt.subplots(3, 1, figsize = (10, 20))

ax[0].scatter(x = proc_df_diamonds['x'], y = proc_df_diamonds['price'])
ax[1].scatter(x = proc_df_diamonds['y'], y = proc_df_diamonds['price'])
ax[2].scatter(x = proc_df_diamonds['z'], y = proc_df_diamonds['price'])

ax[0].set_xlabel("x")
ax[1].set_xlabel("y")
ax[2].set_xlabel("z")

for i in range(3):
    ax[i].set_ylabel("Price")
cond = (proc_df_diamonds['y'] > 30) | (proc_df_diamonds['z'] > 30) | (proc_df_diamonds['z'] < 1.9) | (proc_df_diamonds['x'] == 0) | (proc_df_diamonds['y'] == 0) | (proc_df_diamonds['z'] == 0)  
outliers = proc_df_diamonds[cond]
zero_cond = (proc_df_diamonds['x'] == 0) | (proc_df_diamonds['y'] == 0) | (proc_df_diamonds['z'] == 0)
zero_outliers = proc_df_diamonds[zero_cond]
outliers
zero_outliers
fig, ax = plt.subplots(3, 1, figsize = (10, 20))

ax[0].scatter(x = proc_df_diamonds['x'], y = proc_df_diamonds['price'])
ax[1].scatter(x = proc_df_diamonds['y'], y = proc_df_diamonds['price'])
ax[2].scatter(x = proc_df_diamonds['z'], y = proc_df_diamonds['price'])

ax[0].scatter(x = outliers['x'], y = outliers['price'], color = 'r', label = 'Outliers')
ax[1].scatter(x = outliers['y'], y = outliers['price'], color = 'r', label = 'Outliers')
ax[2].scatter(x = outliers['z'], y = outliers['price'], color = 'r', label = 'Outliers')

ax[0].set_xlabel("x")
ax[1].set_xlabel("y")
ax[2].set_xlabel("z")



for i in range(3):
    ax[i].set_ylabel("Price")
    ax[i].legend(loc = 'lower right')
    ax[i].grid(True)
fig, ax = plt.subplots(3, 1, figsize = (10, 20))

ax[0].scatter(x = proc_df_diamonds['x'], y = proc_df_diamonds['price'])
ax[1].scatter(x = proc_df_diamonds['y'], y = proc_df_diamonds['price'])
ax[2].scatter(x = proc_df_diamonds['z'], y = proc_df_diamonds['price'])

ax[0].scatter(x = zero_outliers['x'], y = zero_outliers['price'], color = 'r', label = 'Zero Outliers')
ax[1].scatter(x = zero_outliers['y'], y = zero_outliers['price'], color = 'r', label = 'Zero Outliers')
ax[2].scatter(x = zero_outliers['z'], y = zero_outliers['price'], color = 'r', label = 'Zero Outliers')

ax[0].set_xlabel("x")
ax[1].set_xlabel("y")
ax[2].set_xlabel("z")

for i in range(3):
    ax[i].set_ylabel("Price")
    ax[i].legend(loc = 'lower right')
    ax[i].grid(True)
removed_df = pd.merge(proc_df_diamonds, outliers, how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)
fig, ax = plt.subplots(3, 1, figsize = (10, 20))

ax[0].scatter(x = removed_df['x'], y = removed_df['price'])
ax[1].scatter(x = removed_df['y'], y = removed_df['price'])
ax[2].scatter(x = removed_df['z'], y = removed_df['price'])

ax[0].set_xlabel("x")
ax[1].set_xlabel("y")
ax[2].set_xlabel("z")

for i in range(3):
    ax[i].set_ylabel("Price")
fig, ax = plt.subplots(3, 1, figsize = (10, 20))

ax[0].scatter(x = removed_df['carat'], y = removed_df['price'])
ax[1].scatter(x = removed_df['depth'], y = removed_df['price'])
ax[2].scatter(x = removed_df['table'], y = removed_df['price'])

ax[0].set_xlabel("carat")
ax[1].set_xlabel("depth")
ax[2].set_xlabel("table")

for i in range(3):
    ax[i].set_ylabel("Price")
cond = (removed_df['carat'] > 3.5) | (removed_df['depth'] > 75) | (removed_df['depth'] < 50) | (removed_df['table'] > 70) | (removed_df['table'] < 50 )
outliers = removed_df[cond]
fig, ax = plt.subplots(3, 1, figsize = (10, 20))

ax[0].scatter(x = removed_df['carat'], y = removed_df['price'])
ax[1].scatter(x = removed_df['depth'], y = removed_df['price'])
ax[2].scatter(x = removed_df['table'], y = removed_df['price'])

ax[0].scatter(x = outliers['carat'], y = outliers['price'], color = 'r', label = 'Outliers')
ax[1].scatter(x = outliers['depth'], y = outliers['price'], color = 'r', label = 'Outliers')
ax[2].scatter(x = outliers['table'], y = outliers['price'], color = 'r', label = 'Outliers')

ax[0].set_xlabel("carat")
ax[1].set_xlabel("depth")
ax[2].set_xlabel("table")

for i in range(3):
    ax[i].set_ylabel("Price")
    ax[i].legend(loc = 'lower right')
    ax[i].grid(True)
cleaned_df = pd.merge(removed_df, outliers, how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)
fig, ax = plt.subplots(3, 1, figsize = (10, 20))

ax[0].scatter(x = removed_df['carat'], y = removed_df['price'])
ax[1].scatter(x = removed_df['depth'], y = removed_df['price'])
ax[2].scatter(x = removed_df['table'], y = removed_df['price'])

ax[0].set_xlabel("carat")
ax[1].set_xlabel("depth")
ax[2].set_xlabel("table")

for i in range(3):
    ax[i].set_ylabel("Price")
    ax[i].grid(True)
#sns.pairplot(removed_df, diag_kws=dict(bins=8))
X = removed_df.drop(columns = ['price'])
Y = removed_df['price']
X.info()
len(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)
minmax_X_test = (X_test - X_train.min()) / (X_train.max() - X_train.min())
minmax_X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
std_X_test = (X_test - X_train.mean()) / X_train.std()
std_X_train = (X_train - X_train.mean()) / X_train.std()
minmax_X_test.describe()
#For this train test set, there is unseen values for trained model which is min of depth which <0
#and max of z which is > 1
minmax_X_train.describe()
lr = LinearRegression(n_jobs = -1)
lr.fit(X_train, Y_train)
train_mse = mean_squared_error(Y_train,lr.predict(X_train))
test_mse = mean_squared_error(Y_test, lr.predict(X_test))
print(train_mse,test_mse)
lr = LinearRegression()
lr.fit(minmax_X_train, Y_train)
minmax_train_mse = mean_squared_error(Y_train,lr.predict(minmax_X_train))
minmax_test_mse = mean_squared_error(Y_test, lr.predict(minmax_X_test))
print(minmax_train_mse, minmax_test_mse)
lr = LinearRegression()
lr.fit(std_X_train, Y_train)
std_train_mse = mean_squared_error(Y_train,lr.predict(std_X_train))
std_test_mse = mean_squared_error(Y_test, lr.predict(std_X_test))
print(std_train_mse, std_test_mse)
def gauge_linear_methods(methods  = "ElasticNet", deg = 1):
    train_mse, test_mse = [], []
    minmax_train_mse, minmax_test_mse = [], []
    std_train_mse, std_test_mse = [], []


    for i in np.geomspace(1e-3, 1e2, 6):
        if methods == "Ridge":
            model = Ridge(alpha = i, random_state = 1, max_iter = 10000)
        elif methods == "Lasso":
            model = Lasso(alpha = i, random_state = 1, max_iter = 10000)
        else:
            model = ElasticNet(alpha = i, random_state = 1, max_iter = 100000)
            
        poly = PolynomialFeatures(degree = deg, include_bias = False)
            
        
        model.fit(poly.fit_transform(X_train), Y_train)
        train_mse.append(mean_squared_error(Y_train, model.predict(poly.fit_transform(X_train))))
        test_mse.append(mean_squared_error(Y_test, model.predict(poly.fit_transform(X_test))))

        model.fit(poly.fit_transform(minmax_X_train), Y_train)
        minmax_train_mse.append(mean_squared_error(Y_train, model.predict(poly.fit_transform(minmax_X_train))))
        minmax_test_mse.append(mean_squared_error(Y_test, model.predict(poly.fit_transform(minmax_X_test))))

        model.fit(poly.fit_transform(std_X_train), Y_train)
        std_train_mse.append(mean_squared_error(Y_train, model.predict(poly.fit_transform(std_X_train))))
        std_test_mse.append(mean_squared_error(Y_test, model.predict(poly.fit_transform(std_X_test))))
        
    plt.figure(figsize = (10,6))
    plt.plot(np.geomspace(1e-3, 1e2, 6), train_mse, marker = 'o', label = "train rmse", drawstyle="steps-post")
    plt.plot(np.geomspace(1e-3, 1e2, 6), test_mse, marker = 'o', label = "test rmse", drawstyle="steps-post")
    plt.plot(np.geomspace(1e-3, 1e2, 6), minmax_train_mse, marker = 'o', label = "minmax train rmse", drawstyle="steps-post")
    plt.plot(np.geomspace(1e-3, 1e2, 6), minmax_test_mse, marker = 'o', label = "minmax test rmse", drawstyle="steps-post")
    plt.plot(np.geomspace(1e-3, 1e2, 6), std_train_mse, marker = 'o', label = "std train rmse", drawstyle="steps-post")
    plt.plot(np.geomspace(1e-3, 1e2, 6), std_test_mse, marker = 'o', label = "std test rmse", drawstyle="steps-post")
    plt.legend(loc = 'upper left')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    print(min(train_mse), min(test_mse))
    print(min(minmax_train_mse), min(minmax_test_mse))
    print(min(std_train_mse), min(std_test_mse))

    return
gauge_linear_methods("Ridge")
gauge_linear_methods("ElasticNet")
gauge_linear_methods("Lasso")
def KNN():
    train_mse, test_mse = [], []
    minmax_train_mse, minmax_test_mse = [], []
    std_train_mse, std_test_mse = [], []


    x_values = list(range(1, 20))
    
    for n in x_values:
        model = KNeighborsRegressor(n_neighbors = n, n_jobs = -1)
            
        model.fit(X_train, Y_train)
        train_mse.append(mean_squared_error(Y_train, model.predict(X_train)))
        test_mse.append(mean_squared_error(Y_test, model.predict(X_test)))

        model.fit(minmax_X_train, Y_train)
        minmax_train_mse.append(mean_squared_error(Y_train, model.predict(minmax_X_train)))
        minmax_test_mse.append(mean_squared_error(Y_test, model.predict(minmax_X_test)))

        model.fit(std_X_train, Y_train)
        std_train_mse.append(mean_squared_error(Y_train, model.predict(std_X_train)))
        std_test_mse.append(mean_squared_error(Y_test, model.predict(std_X_test)))
        
    plt.figure(figsize = (10,6))
    plt.plot(x_values, train_mse, marker = 'o', label = "train rmse", drawstyle="steps-post")
    plt.plot(x_values, test_mse, marker = 'o', label = "test rmse", drawstyle="steps-post")
    plt.plot(x_values, minmax_train_mse, marker = 'o', label = "minmax train rmse", drawstyle="steps-post")
    plt.plot(x_values, minmax_test_mse, marker = 'o', label = "minmax test rmse", drawstyle="steps-post")
    plt.plot(x_values, std_train_mse, marker = 'o', label = "std train rmse", drawstyle="steps-post")
    plt.plot(x_values, std_test_mse, marker = 'o', label = "std test rmse", drawstyle="steps-post")
    plt.legend(loc = 'lower right')
    plt.xlabel('n')
    plt.ylabel('RMSE')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    print('{}: '.format(str(x_values[train_mse.index(min(train_mse))])), min(train_mse), 
          '{}: '.format(str(x_values[test_mse.index(min(test_mse))])),min(test_mse))
    print('{}: '.format(str(x_values[minmax_train_mse.index(min(minmax_train_mse))])), min(minmax_train_mse), 
          '{}: '.format(str(x_values[minmax_test_mse.index(min(minmax_test_mse))])), min(minmax_test_mse))
    print('{}: '.format(str(x_values[std_train_mse.index(min(std_train_mse))])), min(std_train_mse), 
          '{}: '.format(str(x_values[std_test_mse.index(min(std_test_mse))])), min(std_test_mse))

    return
KNN()
poly = PolynomialFeatures(degree = 2, include_bias = False)
lr = LinearRegression(n_jobs = -1)
lr.fit(poly.fit_transform(X_train), Y_train)
poly_train_mse = mean_squared_error(Y_train,lr.predict(poly.fit_transform(X_train)))
poly_test_mse = mean_squared_error(Y_test, lr.predict(poly.fit_transform(X_test)))
print(poly_train_mse, poly_test_mse)
lr.fit(X_train, Y_train)
train_mse = mean_squared_error(Y_train,lr.predict(X_train))
test_mse = mean_squared_error(Y_test, lr.predict(X_test))
print(train_mse,test_mse)
(test_mse - poly_test_mse) / test_mse
gauge_linear_methods("Ridge",deg = 2)
tree = DecisionTreeRegressor(random_state = 1)
tree.fit(X_train, Y_train)
train_mse = mean_squared_error(Y_train,tree.predict(X_train))
test_mse = mean_squared_error(Y_test, tree.predict(X_test))
print(train_mse,test_mse)
Importance = pd.DataFrame({'Importance':tree.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='b', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
tree.fit(minmax_X_train, Y_train)
train_mse = mean_squared_error(Y_train,tree.predict(minmax_X_train))
test_mse = mean_squared_error(Y_test, tree.predict(minmax_X_test))
print(train_mse,test_mse)
tree.fit(std_X_train, Y_train)
train_mse = mean_squared_error(Y_train,tree.predict(std_X_train))
test_mse = mean_squared_error(Y_test, tree.predict(std_X_test))
print(train_mse,test_mse)
path = tree.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
arr = np.array(list(set(ccp_alphas)))
ccp_alphas = arr[arr >= 0]
ccp_alphas = np.geomspace(np.min(ccp_alphas) + 1, np.max(ccp_alphas), 20)
ccp_alphas
clfs = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, Y_train)
    clfs.append(clf)
    
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
#we remove last pruned tree with only one terminal node

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1, figsize = (10,10))
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
for i in range(2):
    ax[i].set_xscale('log')
fig.tight_layout()
train_scores = [mean_squared_error(Y_train,clf.predict(X_train)) for clf in clfs]
test_scores = [mean_squared_error(Y_test,clf.predict(X_test)) for clf in clfs]

fig, ax = plt.subplots(figsize=(10,6))
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.set_xscale("log")
ax.legend()
plt.show()
dic = dict(zip(ccp_alphas, test_scores))
alpha = min(dic, key = dic.get)
alpha, min(test_scores)
rf = RandomForestRegressor(random_state = 1, n_jobs = -1)
rf.fit(X_train, np.ravel(Y_train));
Importance = pd.DataFrame({'Importance':rf.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh' )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
data = {k : [] for k in np.geomspace(1e-3, 1e4, 20)}

for i in list(data.keys()):
    rf = RandomForestRegressor(random_state = 1, n_jobs = -1, ccp_alpha = i)
    rf.fit(X_train, Y_train)

    data[i].append(mean_squared_error(Y_train, rf.predict(X_train)))
    data[i].append(mean_squared_error(Y_test, rf.predict(X_test)))
plt.figure(figsize = (15,15))
plt.plot(list(data.keys()), [i[0] for i in data.values()], marker='o', label="Train_accuracy",
        drawstyle="steps-post")
plt.plot(list(data.keys()), [i[1] for i in data.values()], marker='o', label="Test_accuracy",
        drawstyle="steps-post")

plt.xscale("log")

plt.xlabel("alpha")

plt.legend()

plt.show()
min(data.values())
param = {'objective':'reg:squarederror', 'booster':'gbtree', 'learning_rate' : 0.5,
         'reg_alpha': 10, 'reg_lambda': 10, 'random_state' : 1,
         'n_jobs' : -1}
xgb = XGBRegressor(**param)
xgb.fit(X_train, Y_train);
train_mse = mean_squared_error(Y_train, xgb.predict(X_train))
test_mse = mean_squared_error(Y_test, xgb.predict(X_test))
train_mse, test_mse
#Without fine tuning parameters, XGBoost Yields low RMSE for both training and testing