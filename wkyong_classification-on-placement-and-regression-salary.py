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
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, t
import os

from statsmodels.stats.proportion import proportion_confint

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures

%matplotlib inline
plt.style.use("seaborn-darkgrid")
os.chdir("/kaggle/input")
df_placement = pd.read_csv("factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df_placement.head()
df_placement.info()
df_1 = df_placement.drop(columns = ["salary", 'hsc_s', 'degree_t'])
df_1['gender'].replace(['M', 'F'], [0, 1], inplace = True)
df_1['status'].replace(['Not Placed', 'Placed'], [0, 1], inplace = True)
df_1['ssc_b'].replace(['Central', 'Others'], [0, 1], inplace = True)
df_1['hsc_b'].replace(['Central', 'Others'], [0, 1], inplace = True)
df_1['workex'].replace(['No', 'Yes'], [0, 1], inplace = True)
df_1['specialisation'].replace(['Mkt&Fin', 'Mkt&HR'], [0, 1], inplace = True)
df_1 = df_1.join([pd.get_dummies(df_placement.hsc_s, prefix = df_placement.hsc_s.name),
                  pd.get_dummies(df_placement.degree_t, prefix = df_placement.degree_t.name)
                 ])
#a loop with if condition can also convert to one hot vectors
#can be packed into preprocessing data function
df_1.head()
df_1.info()
plt.figure(figsize = (15,15))
sns.heatmap(df_1.corr(), annot = True)
sns.pairplot(df_1)
X = df_1.drop(columns = ["status"])
Y = df_1['status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)
data = {k : [] for k in range(1, 11)}
#One could choose to express in the form of 4 lists or 4 dictinaries

for i in range(1, 11):
    knncls = KNeighborsClassifier(n_neighbors = i)
    knncls.fit(X_train, Y_train)

    data[i].append(accuracy_score(Y_train, knncls.predict(X_train)))
    data[i].append(accuracy_score(Y_test, knncls.predict(X_test)))
plt.figure(figsize = (15,15))
plt.plot(list(data.keys()), [i[0] for i in data.values()], marker='o', label="Train_accuracy",
        drawstyle="steps-post")
plt.plot(list(data.keys()), [i[1] for i in data.values()], marker='o', label="Test_accuracy",
        drawstyle="steps-post")
plt.legend()

plt.show()
max([i[-1] for i in data.values()])
list(data.values())
knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(X_train, Y_train)

cm = pd.DataFrame(confusion_matrix(Y_test, knn.predict(X_test)).T, index=['No', 'Yes'], columns=['No', 'Yes'])
cm.index.name = 'Predicted'
cm.columns.name = 'True'
cm
correct = confusion_matrix(Y_test, knn.predict(X_test))[0][0] + confusion_matrix(Y_test, knn.predict(X_test))[1][1]
proportion_confint(correct, len(X_test), method = 'wilson')
lr = LogisticRegression(max_iter = 200, n_jobs = -1)
lr.fit(X_train, Y_train)
train_accuracy = accuracy_score(Y_train, lr.predict(X_train))
test_accuracy = accuracy_score(Y_test, lr.predict(X_test))
print(
     "train_accuracy: ", train_accuracy,
     "\ntest_accuracy: ", test_accuracy,)
correct = confusion_matrix(Y_test, lr.predict(X_test))[0][0] + confusion_matrix(Y_test, lr.predict(X_test))[1][1]
proportion_confint(correct, len(X_test), method = 'wilson')
data = {k : [] for k in np.geomspace(1e-3, 1e2, 10)}
#Same procedure as tuning KNNClassifier, depending or not will implement function

for i in data.keys():
    ridge = RidgeClassifier(alpha = i, normalize = True, random_state = 1)
    ridge.fit(X_train, Y_train)
    
    data[i].append(accuracy_score(Y_train, ridge.predict(X_train)))
    data[i].append(accuracy_score(Y_test, ridge.predict(X_test)))
plt.figure(figsize = (10,10))
plt.plot(list(data.keys()), [i[0] for i in data.values()], marker='o', label="Train_accuracy",
        drawstyle="steps-post")
plt.plot(list(data.keys()), [i[1] for i in data.values()], marker='o', label="Test_accuracy",
        drawstyle="steps-post")

plt.xscale("log")


plt.xlabel("alpha")

plt.legend()


plt.show()
max([i[-1] for i in data.values()])
[i[-1] for i in data.values()]
test_alpha = list(data.keys())[4]
ridge = RidgeClassifier(alpha = test_alpha, normalize = True)
ridge.fit(X_train, Y_train)
print(classification_report(Y_test, ridge.predict(X_test)))
cm = pd.DataFrame(confusion_matrix(Y_test, ridge.predict(X_test)).T, index=['No', 'Yes'], columns=['No', 'Yes'])
cm.index.name = 'Predicted'
cm.columns.name = 'True'
cm
mean_squared_error(Y_test, ridge.predict(X_test))
correct = confusion_matrix(Y_test, ridge.predict(X_test))[0][0] + confusion_matrix(Y_test, ridge.predict(X_test))[1][1]
proportion_confint(correct, len(X_test), method = 'wilson')
train_ac, test_ac = [], []

for i in range(1,1001):
    X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X, Y)
    
    ridge_ = RidgeClassifier(alpha = test_alpha, normalize = True)
    ridge_.fit(X_train_, Y_train_)
    
    train_ac.append(accuracy_score(Y_train_, ridge_.predict(X_train_)))
    test_ac.append(accuracy_score(Y_test_, ridge_.predict(X_test_)))
    
plt.figure(figsize = (10,6))
#plt.plot([i for i in range(1,1001)], train_ac, marker='o', label="Train_accuracy",
#        drawstyle="steps-post")
plt.plot([i for i in range(1,1001)], test_ac, marker='o', label="Test_accuracy",
        drawstyle="steps-post")

plt.legend()


plt.show()
sns.distplot(test_ac, bins= 20)
shapiro(test_ac)
#default tree
tree = DecisionTreeClassifier(random_state = 0)
tree.fit(X_train, Y_train)
print(classification_report(Y_test, tree.predict(X_test)))
mean_squared_error(Y_test, tree.predict(X_test)), accuracy_score(Y_test, tree.predict(X_test))
fig, ax = plt.subplots(figsize=(100, 100))
features = X_train.columns

plot_tree(tree, filled = True, ax = ax, feature_names = features, proportion = True, rounded = True);
Importance = pd.DataFrame({'Importance':tree.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='b', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
path = tree.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
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
fig.tight_layout()
train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]

fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()
dic = dict(zip(ccp_alphas, test_scores))
alpha = max(dic, key = dic.get)
tree = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
tree.fit(X_train, Y_train)
print(classification_report(Y_test, tree.predict(X_test)))
fig, ax = plt.subplots(figsize=(100, 100))
features = X_train.columns

plot_tree(tree, filled = True, ax = ax, feature_names = features, proportion = True, rounded = True);
rf = RandomForestClassifier(random_state = 1, n_jobs = -1)
rf.fit(X_train, np.ravel(Y_train));
Importance = pd.DataFrame({'Importance':rf.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh' )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
data = {k : [] for k in np.geomspace(1e-3, 1e2, 20)}
#Same procedure as tuning KNNClassifier, depending or not will implement function

for i in list(data.keys()):
    rf = RandomForestClassifier(random_state = 1, n_jobs = -1, ccp_alpha = i)
    rf.fit(X_train, Y_train)

    data[i].append(accuracy_score(Y_train, rf.predict(X_train)))
    data[i].append(accuracy_score(Y_test, rf.predict(X_test)))
plt.figure(figsize = (15,15))
plt.plot(list(data.keys()), [i[0] for i in data.values()], marker='o', label="Train_accuracy",
        drawstyle="steps-post")
plt.plot(list(data.keys()), [i[1] for i in data.values()], marker='o', label="Test_accuracy",
        drawstyle="steps-post")

plt.xscale("log")

plt.xlabel("alpha")

plt.legend()

plt.show()
regr = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.01, random_state = 1)
regr.fit(X_train, np.ravel(Y_train))
Importance = pd.DataFrame({'Importance':regr.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
grid_params = {'learning_rate': np.geomspace(1e-4, 100, num = 7),
               'n_estimators': np.arange(100, 1000, 100),
               'max_depth': [i for i in range(1,11)]}

gs = GridSearchCV(regr, grid_params, cv = 10, n_jobs = -1, scoring = 'accuracy')
gs.fit(X_train, Y_train)
gs.best_estimator_.get_params()
gs.best_estimator_.fit(X_train, Y_train)
accuracy_score(Y_test, gs.best_estimator_.predict(X_test))
data = {k : [] for k in np.geomspace(1e-4, 100, num = 7)}
#Same procedure as tuning KNNClassifier, depending or not will implement function
#Different learning rate

for i in list(data.keys()):
    regr = GradientBoostingClassifier(n_estimators = 1000, learning_rate = i, random_state = 1)
    regr.fit(X_train, Y_train)

    data[i].append(accuracy_score(Y_train, regr.predict(X_train)))
    data[i].append(accuracy_score(Y_test, regr.predict(X_test)))
plt.figure(figsize = (10,6))
plt.plot(list(data.keys()), [i[0] for i in data.values()], marker='o', label="Train_accuracy",
        drawstyle="steps-post")
plt.plot(list(data.keys()), [i[1] for i in data.values()], marker='o', label="Test_accuracy",
        drawstyle="steps-post")

plt.xscale("log")

plt.xlabel("alpha")

plt.legend()

plt.show()
depth = [i for i in range(1,11)]


for i in depth:
    
    regr = GradientBoostingClassifier(n_estimators = 500, learning_rate = 10, random_state = 1, max_depth = i)
    #Using prior best test accuracy learning rate
    regr.fit(X_train, Y_train)

    train_acc.append(accuracy_score(Y_train, regr.predict(X_train)))
    test_acc.append(accuracy_score(Y_test, regr.predict(X_test)))
plt.figure(figsize = (10,6))
plt.plot(depth, train_acc, marker='o', label="Train_accuracy",
        drawstyle="steps-post")
plt.plot(depth, test_acc, marker='o', label="Test_accuracy",
        drawstyle="steps-post")

plt.xlabel("depth")

plt.legend()

plt.show()
df_placement.info()
df_2 = df_placement.drop(columns = ["status", 'hsc_s', 'degree_t'])
df_2 = df_2.dropna()
df_2['gender'].replace(['M', 'F'], [0, 1], inplace = True)
df_2['ssc_b'].replace(['Central', 'Others'], [0, 1], inplace = True)
df_2['hsc_b'].replace(['Central', 'Others'], [0, 1], inplace = True)
df_2['workex'].replace(['No', 'Yes'], [0, 1], inplace = True)
df_2['specialisation'].replace(['Mkt&Fin', 'Mkt&HR'], [0, 1], inplace = True)
df_2 = df_2.join([pd.get_dummies(df_placement.hsc_s, prefix = df_placement.hsc_s.name),
                  pd.get_dummies(df_placement.degree_t, prefix = df_placement.degree_t.name)
                 ])
#a loop with if condition can also convert to one hot vectors
#can be packed into preprocessing data function
df_2.head()
plt.figure(figsize = (15,15))
sns.heatmap(df_2.corr(method = "spearman"), annot = True)
X = df_2.drop(columns = ["salary"])
Y = df_2["salary"]
reg_X_train, reg_X_test, reg_Y_train, reg_Y_test = train_test_split(X, Y, random_state = 1) 
lin_r = LinearRegression(normalize = True, n_jobs = -1)
lin_r.fit(reg_X_train, reg_Y_train)
def scoring(model, Y_test = reg_Y_test, X_test = reg_X_test, Y_train = reg_Y_train, X_train = reg_X_train):
    R_squared = model.score(X_test, Y_test)
    R2_adj =1 - (((1 - R_squared) * (len(X_test) - 1)) / (len(X_test) - (len(X_test.columns) - 1)))
    test_mse = mean_squared_error(Y_test, model.predict(X_test))
    train_mse = mean_squared_error(Y_train, lin_r.predict(X_train))
    return [R_squared, R2_adj, train_mse, test_mse] 
    
scoring(lin_r)
lasso_scores, ridge_scores = [], []

for i in np.geomspace(1e-2, 1e2, 5):
    lasso = Lasso(alpha = i, normalize = True, random_state = 1, max_iter = 50000)
    ridge = Ridge(alpha = i, normalize = True, max_iter = 50000)
    
    lasso.fit(reg_X_train, reg_Y_train)
    ridge.fit(reg_X_train, reg_Y_train)
    
    lasso_scores.append(scoring(lasso))
    ridge_scores.append(scoring(ridge))
plt.figure(figsize = (10,6))

plt.plot(np.geomspace(1e-2, 1e2, 5), [i[0] for i in lasso_scores], label = "Lasso R2", drawstyle="steps-post")
plt.plot(np.geomspace(1e-2, 1e2, 5), [i[0] for i in ridge_scores], label = "Ridge R2", drawstyle="steps-post")
plt.xscale("log")
plt.legend()

plt.show()

plt.figure(figsize = (10,6))

plt.plot(np.geomspace(1e-2, 1e2, 5), [i[1] for i in lasso_scores], label = "Lasso Adjusted R2", drawstyle="steps-post")
plt.plot(np.geomspace(1e-2, 1e2, 5), [i[1] for i in ridge_scores], label = "Ridge Adjusted R2", drawstyle="steps-post")
plt.xscale("log")
plt.legend()
plt.figure(figsize = (10,6))

plt.plot(np.geomspace(1e-2, 1e2, 5), [i[3] for i in ridge_scores],
         label = "Ridge test mse", drawstyle="steps-post")
plt.plot(np.geomspace(1e-2, 1e2, 5), [i[3] for i in lasso_scores],
         label = "Lasso test mse", drawstyle="steps-post")

plt.xscale("log")
plt.legend()

plt.show()
reg_tree = DecisionTreeRegressor(random_state = 1)
reg_tree.fit(reg_X_train, reg_Y_train)
scoring(reg_tree)
#Very poor values of R2
path = reg_tree.cost_complexity_pruning_path(reg_X_train, reg_Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots(figsize = [10,6])
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=1, ccp_alpha=ccp_alpha)
    clf.fit(reg_X_train, reg_Y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))
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
fig.tight_layout()
scores = [scoring(clf) for clf in clfs]

fig, ax = plt.subplots(2, 1, figsize=(10,6))
ax[0].set_xlabel("alpha")
ax[1].set_xlabel("alpha")
ax[0].set_ylabel("mse")
ax[1].set_ylabel("mse")
ax[0].set_title("MSE vs alpha for training and testing sets")
ax[0].plot(ccp_alphas, [i[2] for i in scores], marker='o', label="train mse",
        drawstyle="steps-post")
ax[1].plot(ccp_alphas, [i[3] for i in scores], marker='o', label="test mse",
        drawstyle="steps-post")
ax[0].legend()
ax[1].legend()
plt.show()