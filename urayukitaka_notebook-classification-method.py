# Basic Libraries

import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Visualization

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

import seaborn as sns



# Data preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# StratifiedKFold

from sklearn.model_selection import StratifiedKFold



# Grid search

from sklearn.model_selection import GridSearchCV



# Learning curve

from sklearn.model_selection import learning_curve



# Validation curve

from sklearn.model_selection import validation_curve

from sklearn.model_selection import cross_val_score



# Confusion matrix and scores

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



# ROC curve

from sklearn.metrics import roc_curve, auc

from scipy import interp
class k_fold_cross_val:

    def __init__(self, X_train, y_train, estimator, cv):

        self.X_train = X_train

        self.y_train = y_train

        self.estimator = estimator

        self.cv = cv

        

    def cross_val_kfold(self):

        kfold = StratifiedKFold(n_splits=self.cv, random_state=10)

        self.kfold = kfold

        

        scores = []

        for train_idx, test_idx in self.kfold.split(self.X_train, self.y_train):

            self.estimator.fit(self.X_train[train_idx], self.y_train.values[train_idx])

            score = self.estimator.score(self.X_train[test_idx], self.y_train.values[test_idx])

            scores.append(score)

            print("Class: %s, Acc: %.3f" % (np.bincount(self.y_train.values[train_idx]), score))

            self.scores = scores

            

    def score(self):

        scores = cross_val_score(estimator=self.estimator, X=self.X_train, y=self.y_train, cv=self.cv, n_jobs=1)

        print("CV accuracy scores: %s" % self.scores)

        print("CV accuracy: %.3f +/- %.3f" % (np.mean(self.scores), np.std(self.scores)))

        

    def draw_roc_curve(self, X_test, y_test):

        self.X_test = X_test

        self.y_test = y_test

        

        mean_tpr=0

        mean_fpr=np.linspace(0,1,100)

        plt.figure(figsize=(10,6))

        for train_idx, test_idx in self.kfold.split(self.X_train, self.y_train):

            proba = self.estimator.fit(self.X_train[train_idx], self.y_train.values[train_idx]).predict_proba(self.X_train[test_idx])

            fpr, tpr, thresholds = roc_curve(y_true=self.y_train.values[test_idx], y_score=proba[:,1], pos_label=1)

            mean_tpr += interp(mean_fpr, fpr, tpr)

            mean_tpr[0] = 0

            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=1, label="ROC fold (area=%.2f)" %(roc_auc))

        

        # Line

        plt.plot([0,1], [0,1], linestyle='--', color=(0.6,0.6,0.6), label="random guessing")

        # plot mean of fpr, tpr roc_auc

        mean_tpr /= self.cv

        mean_tpr[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tpr)

        plt.plot(mean_fpr, mean_tpr, 'k--', label="mean ROC (area = %.2f)" % mean_auc, color="blue")

        # Line

        plt.plot([0,0,1], [0,1,1], lw=2, linestyle=':', color="black", label='perfect performance')

        plt.xlabel("false positive rate")

        plt.ylabel("true positive rate")

        plt.title("Receiver Operator Characteristic")

        plt.legend()
def draw_learning_curve(estimator, X_train, y_train):

    # learning curve

    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X_train, y=y_train, train_sizes=np.linspace(0.1,1,10), cv=10, n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)



    # plot

    plt.figure(figsize=(10,6))

    # train data

    plt.plot(train_sizes, train_mean, color="blue", marker='o', markersize=5, label='training accuracy')

    plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, color="blue", alpha=0.15)

    # val data

    plt.plot(train_sizes, test_mean, color="green", marker='s', linestyle='--', markersize=5, label='validation accuracy')

    plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, color="green", alpha=0.15)



    plt.grid()

    plt.xlabel("Number of trainig samples")

    plt.ylabel("Accuracy")

    plt.ylim([0.8,1.0])

    plt.title("Learning curve")

    plt.legend()
def draw_validation_curve(estimator, X_train, y_train, param_name, param_range, xscale):

    # validation curve

    train_scores, test_scores = validation_curve(estimator=estimator, X=X_train, y=y_train, param_name=param_name, param_range=param_range, cv=10)

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)

    

    # plot

    plt.figure(figsize=(10,6))

    # train data

    plt.plot(param_range, train_mean, color="blue", marker='o', markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean+train_std, train_mean-train_std, color="blue", alpha=0.15)

    # val data

    plt.plot(param_range, test_mean, color="green", marker='s', linestyle='--', markersize=5, label='validation accuracy')

    plt.fill_between(param_range, test_mean+test_std, test_mean-test_std, color="green", alpha=0.15)



    plt.grid()

    plt.xlabel("{}".format(param_name))

    if xscale=="log":

        plt.xscale("log")

    else:

        pass

    plt.ylabel("Accuracy")

    plt.ylim([0.8,1.0])

    plt.title("Validation curve")

    plt.legend()
def confmat_roccurve(X_test, y_test, y_pred, estimator):

    # create confusion matrix

    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

    # visualiazation confusion matrix

    fig, ax = plt.subplots(1,2,figsize=(18,6))

    

    ax[0].matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

    for i in range(confmat.shape[0]):

        for j in range(confmat.shape[1]):

            ax[0].text(x=j, y=i, s=confmat[i,j], va="center", ha="center")

            

    ax[0].set_xlabel("predicted label")

    ax[0].set_ylabel("true label")

    ax[0].set_title("confusion matrix")

    # Score

    print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

    print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

    print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

    print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

    

    # visualization roc curve

    y_score = estimator.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)

    ax[1].plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr), color="blue")

    ax[1].plot([0,1], [0,1], linestyle='--', color=(0.6,0.6,0.6), label='random')

    ax[1].plot([0,0,1], [0,1,1], linestyle=':', color="black", label='perfect performance')

    ax[1].set_xlabel("false positive rate")

    ax[1].set_ylabel("true positive rate")

    ax[1].set_title("Receiver Operator Characteristic")

    ax[1].legend()
## Data loading

df = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv", header=0)
# data frame

df.head()
# Null values

df.isnull().sum().sum()
# Data info

df.info()
plt.figure(figsize=(10,6))

sns.distplot(df["price"])

plt.vlines([df["price"].quantile(0.75)], 0, 0.000002, "red", linestyles='-')

plt.vlines([df["price"].quantile(0.98)], 0, 0.000002, "blue", linestyles='-')

plt.xlabel("price")

plt.ylabel("frequency")
quat_98 = df["price"].quantile(0.98)

quat_75 = df["price"].quantile(0.75)



df = df[df["price"]<=quat_98]
# define function

def price_flg(x):

    if x["price"] > quat_75:

        res = 1

    else:

        res = 0

    return res

# apply function

df["price_flg"] = df.apply(price_flg, axis=1)
# Checking

df["price_flg"].value_counts()
# Target value

y = df["price_flg"]
ex_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',

              'condition', 'grade','sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
# yr_built

latest_year = df["yr_built"].max()

df["yr_built"] = latest_year - df["yr_built"]
# define function

def renov(x):

    if x["yr_renovated"] == 0:

        res = x["yr_built"]

    else:

        res = latest_year - x["yr_renovated"]

    return res



# apply function

df["yr_renovated"] = df.apply(renov, axis=1)
X = df[ex_columns]
# Sample 200

sns.pairplot(X.sample(200))
## Correlation

matrix = X.corr()

plt.figure(figsize=(10,10))

sns.heatmap(matrix, vmax=1, vmin=-1, cmap="bwr", square=True)
# data split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# Scaling

sc = StandardScaler()

sc.fit(X_train)



X_train_std = sc.fit_transform(X_train)

X_test_std = sc.fit_transform(X_test)
# Library

from sklearn.linear_model import LogisticRegression



# Instance

lr = LogisticRegression()
# prameters

param_range = [0.001, 0.01, 0.1, 1.0, 10, 100]

penalty = ['l1', 'l2']

param_grid = [{"C":param_range, "penalty":penalty}]



# Optimization by Grid search

gs = GridSearchCV(estimator=lr, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)



gs = gs.fit(X_train_std, y_train)



print(gs.best_score_)

print(gs.best_params_)
# Test data validation by best estimator

gs_l = gs.best_estimator_

y_pred = gs_l.predict(X_test_std)

print('Test accuracy: %.3f' % gs_l.score(X_test_std, y_test))
# cross validation

cv = k_fold_cross_val(X_train_std, y_train, gs_l, 5)

cv.cross_val_kfold()
# cross val score

cv.score()
# cv roc curve

cv.draw_roc_curve(X_test_std, y_test)
# learning curve

draw_learning_curve(gs_l, X_train_std, y_train)
# validation curve

# param C

draw_validation_curve(lr, X_train_std, y_train, "C", param_range, "log")
# Confusion matrix and ROC curve

confmat_roccurve(X_test_std, y_test, y_pred, gs_l)
# coefficient

coef = pd.DataFrame({"Variable":X.columns, "Coef":gs_l.coef_[0]}).sort_values(by="Coef")

intercept = pd.DataFrame([["intercept", gs_l.intercept_[0]]], columns=coef.columns)



coef = coef.append(intercept)



# Visualization

plt.figure(figsize=(10,6))

plt.bar(coef["Variable"], coef["Coef"])

plt.xlabel("Variables")

plt.xticks(rotation=90)

plt.ylabel("Coefficient")
# Library

import scipy as sp

from scipy import stats

import statsmodels.formula.api as smf

import statsmodels.api as sm
data = pd.concat([pd.DataFrame(X_train_std, columns=X.columns), pd.DataFrame({"price_flg":y_train.values})], axis=1)

data.head()
# predict logistic regression model

lr_stats = smf.glm(formula="price_flg ~ bedrooms+bathrooms+sqft_living+sqft_lot+floors+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+yr_renovated+sqft_living15+sqft_lot15", data=data, family=sm.families.Binomial()).fit()

lr_stats.summary()
# plot by max coef variable vs prediction

plt.figure(figsize=(10,6))

sns.lmplot(x="grade", y="price_flg", data=data, logistic=True, scatter_kws={"color":"blue"}, line_kws={"color":"black"}, x_jitter=0.1, y_jitter=0.02)
# Library

# Omitted because calculation is heavy

# from sklearn.svm import SVC



# Instance

# svm = SVC(random_state=10, kernel="linear", probability=True)
# prameters

# Omitted because calculation is heavy

# param_range = [0.1, 1.0, 10, 100]

# param_grid = [{"C":param_range, "gamma":param_range}]



# Optimization by Grid search

# gs = GridSearchCV(estimator=svm, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)



# gs = gs.fit(X_train_std, y_train)



# print(gs.best_score_)

# print(gs.best_params_)
# Test data validation by best estimator

# Omitted because calculation is heavy

# gs_sl = gs.best_estimator_

# y_pred = gs_sl.predict(X_test_std)

# print('Test accuracy: %.3f' % gs_sl.score(X_test_std, y_test))
# cross validation

# Omitted because calculation is heavy

# cv = k_fold_cross_val(X_train_std, y_train, gs_sl, 5)

# cv.cross_val_kfold()
# cross val score

# Omitted because calculation is heavy

# cv.score()
# cv roc curve

# Omitted because calculation is heavy

# cv.draw_roc_curve(X_test_std, y_test)
# learning curve

# Omitted because calculation is heavy

# draw_learning_curve(gs_sl, X_train_std, y_train)
# validation curve

# Omitted because calculation is heavy

# draw_validation_curve(svm, X_train_std, y_train, "C", param_range, "log")
# Confusion matrix and ROC curve

# confmat_roccurve(X_test_std, y_test, y_pred, gs_sl)
# Library

# Omitted because calculation is heavy

# from sklearn.svm import SVC



# Instance

# svm = SVC(random_state=10, kernel='rbf', probability=True)
# prameters

# Omitted because calculation is heavy

# param_range = [0.1, 1.0, 10, 100]

# param_grid = [{"C":param_range, "gamma":param_range}]



# Optimization by Grid search

# gs = GridSearchCV(estimator=svm, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)



# gs = gs.fit(X_train_std, y_train)



# print(gs.best_score_)

# print(gs.best_params_)
# Test data validation by best estimator

# Omitted because calculation is heavy

# gs_sr = gs.best_estimator_

# y_pred = gs_sr.predict(X_test_std)

# print('Test accuracy: %.3f' % gs_sr.score(X_test_std, y_test))
# cross validation

# Omitted because calculation is heavy

# cv = k_fold_cross_val(X_train_std, y_train, gs_sr, 5)

# cv.cross_val_kfold()
# cross val score

# Omitted because calculation is heavy

# cv.score()
# cv roc curve

# Omitted because calculation is heavy

# cv.draw_roc_curve(X_test_std, y_test)
# learning curve

# Omitted because calculation is heavy

# draw_learning_curve(gs_s, X_train_std, y_train)
# validation curve

# Omitted because calculation is heavy

# draw_validation_curve(svm, X_train_std, y_train, "C", param_range, "log")
# Confusion matrix and ROC curve

# confmat_roccurve(X_test_std, y_test, y_pred, gs_sr)
# Library

from sklearn.neighbors import KNeighborsClassifier



# Instance

knn = KNeighborsClassifier(metric='minkowski')
# prameters

param_range = [10, 15, 20, 25]

param_grid = [{"n_neighbors":param_range, "p":[1,2]}]



# Optimization by Grid search

gs = GridSearchCV(estimator=knn, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)



gs = gs.fit(X_train_std, y_train)



print(gs.best_score_)

print(gs.best_params_)
# Test data validation by best estimator

gs_kn = gs.best_estimator_

y_pred = gs_kn.predict(X_test_std)

print('Test accuracy: %.3f' % gs_kn.score(X_test_std, y_test))
# cross validation

cv = k_fold_cross_val(X_train_std, y_train, gs_kn, 5)

cv.cross_val_kfold()
# cross val score

cv.score()
# cv roc curve

cv.draw_roc_curve(X_test_std, y_test)
# learning curve

draw_learning_curve(gs_kn, X_train_std, y_train)
# validation curve

draw_validation_curve(knn, X_train_std, y_train, "n_neighbors", param_range, "log")
# Confusion matrix and ROC curve

confmat_roccurve(X_test_std, y_test, y_pred, gs_kn)
# Library

from sklearn.tree import DecisionTreeClassifier



# Instance

tree = DecisionTreeClassifier(random_state=10)
# prameters

param_range = [1, 3, 5, 10]

leaf = [17, 18, 19, 20, 21, 22, 23]

criterion = ["entropy", "gini", "error"]

param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



# Optimization by Grid search

gs = GridSearchCV(estimator=tree, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)



gs = gs.fit(X_train.values, y_train)



print(gs.best_score_)

print(gs.best_params_)
# Test data validation by best estimator

gs_tr = gs.best_estimator_

y_pred = gs_tr.predict(X_test.values)

print('Test accuracy: %.3f' % gs_tr.score(X_test, y_test))
# cross validation

cv = k_fold_cross_val(X_train.values, y_train, gs_tr, 5)

cv.cross_val_kfold()
# cross val score

cv.score()
# cv roc curve

cv.draw_roc_curve(X_test.values, y_test)
# learning curve

draw_learning_curve(gs_tr, X_train, y_train)
# validation curve

draw_validation_curve(tree, X_train, y_train, "max_depth", param_range, "")
# Confusion matrix and ROC curve

confmat_roccurve(X_test, y_test, y_pred, gs_tr)
# Library

!pip install dtreeviz

from sklearn import tree

from dtreeviz.trees import *

import graphviz
# Fitting

tree_c = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=20)

tree_c.fit(X_train, y_train)
# Visualization

viz = dtreeviz(tree_c, X_train, y_train, target_name="price_flg", feature_names=list(X_train.columns), class_names=list(y_train))

viz
# Library

from sklearn.ensemble import RandomForestClassifier



# Instance

forest = RandomForestClassifier(n_estimators=10, random_state=10)
# prameters

param_range = [5, 10, 15, 20]

leaf = [15, 20, 25, 30]

criterion = ["entropy", "gini", "error"]

param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



# Optimization by Grid search

gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)



gs = gs.fit(X_train, y_train)



print(gs.best_score_)

print(gs.best_params_)
# Test data validation by best estimator

gs_rf = gs.best_estimator_

y_pred = gs_rf.predict(X_test)

print('Test accuracy: %.3f' % gs_rf.score(X_test, y_test))
# cross validation

cv = k_fold_cross_val(X_train.values, y_train, gs_rf, 5)

cv.cross_val_kfold()
# cross val score

cv.score()
# cv roc curve

cv.draw_roc_curve(X_test.values, y_test)
# learning curve

draw_learning_curve(gs_rf, X_train, y_train)
# validation curve

draw_validation_curve(forest, X_train, y_train, "max_depth", param_range, "")
# Confusion matrix and ROC curve

confmat_roccurve(X_test, y_test, y_pred, gs_rf)
forest = RandomForestClassifier(criterion='gini', max_depth=10, max_leaf_nodes=19)

forest.fit(X_train, y_train)



importance = forest.feature_importances_



indices = np.argsort(importance)[::-1]



for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" %(f+1, 30, X_train.columns[indices[f]], importance[indices[f]]))
# Library

import xgboost as xgb



# Instance

xgb = xgb.XGBClassifier(random_state=10)
# prameters

max_depth = [10, 15, 20, 25]

min_samples_leaf = [1,3,5]

min_samples_split = [1,2,4]



param_grid = [{"max_depth":max_depth,

               "min_samples_leaf":min_samples_leaf, "min_samples_split":min_samples_split}]



# Optimization by Grid search

gs = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)



gs = gs.fit(X_train, y_train)



print(gs.best_score_)

print(gs.best_params_)
# Test data validation by best estimator

gs_xg = gs.best_estimator_

y_pred = gs_xg.predict(X_test)

print('Test accuracy: %.3f' % gs_xg.score(X_test, y_test))
# cross validation

cv = k_fold_cross_val(X_train.values, y_train, gs_xg, 5)

cv.cross_val_kfold()
# cross val score

cv.score()
# cv roc curve

cv.draw_roc_curve(X_test.values, y_test)
# learning curve

# draw_learning_curve(gs_xg, X_train.values, y_train)
# validation curve

# draw_validation_curve(xgb, X_train.values, y_train, "max_depth", param_range, "")
# Confusion matrix and ROC curve

confmat_roccurve(X_test.values, y_test, y_pred, gs_xg)
# Library

import lightgbm as lgb



# Instance

lgb = lgb.LGBMClassifier()
# prameters

max_depth = [5, 10, 15]

min_samples_leaf = [1,3,5,7]

min_samples_split = [4,6, 8, 10]



param_grid = [{"max_depth":max_depth,

               "min_samples_leaf":min_samples_leaf, "min_samples_split":min_samples_split}]



# Optimization by Grid search

gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)



gs = gs.fit(X_train, y_train)



print(gs.best_score_)

print(gs.best_params_)
# Test data validation by best estimator

gs_lg = gs.best_estimator_

y_pred = gs_lg.predict(X_test)

print('Test accuracy: %.3f' % gs_lg.score(X_test, y_test))
# cross validation

cv = k_fold_cross_val(X_train.values, y_train, gs_lg, 5)

cv.cross_val_kfold()
# cross val score

cv.score()
# cv roc curve

cv.draw_roc_curve(X_test.values, y_test)
# learning curve

draw_learning_curve(gs_lg, X_train, y_train)
# validation curve

draw_validation_curve(lgb, X_train, y_train, "max_depth", param_range, "")
# Confusion matrix and ROC curve

confmat_roccurve(X_test.values, y_test, y_pred, gs_lg)
# ROC AUC scores, calculated from y_pred

lr_score = roc_auc_score(y_true=y_test, y_score=gs_l.predict(X_test_std))

kn_score = roc_auc_score(y_true=y_test, y_score=gs_kn.predict(X_test_std))

tr_score = roc_auc_score(y_true=y_test, y_score=gs_tr.predict(X_test))

rf_score = roc_auc_score(y_true=y_test, y_score=gs_rf.predict(X_test))

xg_score = roc_auc_score(y_true=y_test, y_score=gs_xg.predict(X_test.values))

lg_score = roc_auc_score(y_true=y_test, y_score=gs_lg.predict(X_test))
# scores

name = ["logistic regression", "k Neighbors", "Decision tree", "Random forest", "XGB", "LGBM"]

score = [lr_score, kn_score, tr_score, rf_score, xg_score, lg_score]



last = pd.DataFrame({"name":name, "score":score})



plt.figure(figsize=(10,6))

plt.bar(last["name"], last["score"])

plt.xlabel("Classification method")

plt.ylabel("ROC AUC score")

plt.xticks(rotation=90)