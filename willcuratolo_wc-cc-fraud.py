# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# imported packages

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

import time

import operator

from sklearn import svm

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



# deal with warnings for presentation

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/creditcard.csv")
df.isna().sum()
num_transactions = len(df)

print(num_transactions)
fig, ax = plt.subplots()

sns.distplot(df['Amount'].values, color='b')

ax.set_title('Transaction Amount Distribution', fontsize=14)

ax.set_xlim(df['Amount'].min(),df['Amount'].max())

fig.set_size_inches(8, 6)

print("The mean transaction amount is $" + str(round(df['Amount'].mean()))

      + ", the maximum transaction is $" +  str(df['Amount'].max())

      + ", and \n the minimum is $" + str(df['Amount'].min()))
num_fraud = len(df[df['Class'] == 1])

percent_fraud = num_fraud / num_transactions * 100

print("The number of fraudulent transactions was " + str(num_fraud) + ", which is " +

      str(round(percent_fraud, ndigits=2)) + "% of the total transactions.")

sns.countplot('Class', data=df)

plt.title('Class Distributions \n (0: No Fraud, 1: Fraud)', fontsize=14)

plt.show()
ax = sns.distplot(df['Time'].values, color='r')

ax.set_title('Transaction Time Distribution', fontsize=14)

ax.set_xlim(df['Time'].min(),df['Time'].max())

plt.show()
df.head()
df.describe()
rob_scaler = RobustScaler()



df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))



df.drop(['Time','Amount'], axis=1, inplace=True)
# split the df into fraud and non-fraud

fraud_df = df.loc[df['Class'] == 1]

non_fraud_df = df.loc[df['Class'] == 0]



# choose 492 of the non-fraud transactions

non_fraud_selected_df = non_fraud_df.sample(n = 492)



# concat the fraud and selected non-fraud, then randomize the order

sub_samp_df = pd.concat([fraud_df, non_fraud_selected_df]).sample(frac = 1)



# now plot the distribution again, just to check

sns.countplot('Class', data=sub_samp_df)

plt.title('Class Distributions \n (0: No Fraud, 1: Fraud)')

plt.show()
f, ax = plt.subplots()



sub_samp_corr = sub_samp_df.corr()

sns.heatmap(sub_samp_corr, cmap='coolwarm_r', annot_kws={'size':20})

ax.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)

f.set_size_inches(8, 6)

plt.show()

sub_samp_corr[sub_samp_corr['Class'].abs() > 0.5]['Class']
f, axes = plt.subplots(ncols=2, figsize=(20,10))



# positive correlations 

sns.boxplot(x="Class", y="V4", data=sub_samp_df, ax=axes[0], whis = 2.5)

axes[0].set_title('V4 vs Class Positive Correlation')



sns.boxplot(x="Class", y="V11", data=sub_samp_df, ax=axes[1], whis = 2.5)

axes[1].set_title('V11 vs Class Positive Correlation')





plt.show()
f, axes = plt.subplots(ncols=4, nrows=2, sharex = 'col', figsize=(20,10))



# negative correlations

sns.boxplot(x="Class", y="V3", data=sub_samp_df, ax=axes[0, 0], whis = 2.5)

axes[0, 0].set_title('V3 vs Class Negative Correlation')



sns.boxplot(x="Class", y="V9", data=sub_samp_df, ax=axes[0, 1], whis = 2.5)

axes[0, 1].set_title('V9 vs Class Negative Correlation')



sns.boxplot(x="Class", y="V10", data=sub_samp_df, ax=axes[0, 2], whis = 2.5)

axes[0, 2].set_title('V10 vs Class Negative Correlation')



sns.boxplot(x="Class", y="V12", data=sub_samp_df, ax=axes[0, 3], whis = 2.5)

axes[0, 3].set_title('V12 vs Class Negative Correlation')



sns.boxplot(x="Class", y="V14", data=sub_samp_df, ax=axes[1, 0], whis = 2.5)

axes[1, 0].set_title('V14 vs Class Negative Correlation')



sns.boxplot(x="Class", y="V16", data=sub_samp_df, ax=axes[1, 1], whis = 2.5)

axes[1, 1].set_title('V16 vs Class Negative Correlation')



sns.boxplot(x="Class", y="V17", data=sub_samp_df, ax=axes[1, 2], whis = 2.5)

axes[1, 2].set_title('V17 vs Class Negative Correlation')





plt.show()
# choose iqr ratio

iqr_ratio = 2.5

    

# get v3 values

v3_fraud = sub_samp_df['V3'].loc[sub_samp_df['Class'] == 1].values



# calcualte v3 iqr

q25, q75 = np.percentile(v3_fraud, 25), np.percentile(v3_fraud, 75)

v3_iqr = q75 - q25



# determine outlier cutoffs for v3

v3_cut_off = v3_iqr * iqr_ratio

v3_lower, v3_upper = q25 - v3_cut_off, q75 + v3_cut_off



# get v9 values

v9_fraud = sub_samp_df['V9'].loc[sub_samp_df['Class'] == 1].values



# calcualte v9 iqr

q25, q75 = np.percentile(v9_fraud, 25), np.percentile(v9_fraud, 75)

v9_iqr = q75 - q25



# determine outlier cutoffs for v3

v9_cut_off = v9_iqr * iqr_ratio

v9_lower, v9_upper = q25 - v9_cut_off, q75 + v9_cut_off



# get v10 values

v10_fraud = sub_samp_df['V10'].loc[sub_samp_df['Class'] == 1].values



# calcualte v10 iqr

q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)

v10_iqr = q75 - q25



# determine outlier cutoffs for v10

v10_cut_off = v10_iqr * iqr_ratio

v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off







sub_samp_df = sub_samp_df.drop(sub_samp_df[(sub_samp_df['V3'] > v3_upper) | (sub_samp_df['V3'] < v3_lower) | 

                                          (sub_samp_df['V9'] > v9_upper) | (sub_samp_df['V9'] < v9_lower) |

                                          (sub_samp_df['V10'] > v3_upper) | (sub_samp_df['V10'] < v3_lower)].index)

print("number of dropped rows = " + str((492 * 2) - len(sub_samp_df)))
# preparing the data for classifiers

y = sub_samp_df['Class']

X = sub_samp_df.drop('Class', axis = 1)



# splitting data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)



# formatting them into arrays for the classifiers

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values

cfs = {

    "Logisitic Regression": LogisticRegression(),

    "KNN": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "Decision Tree Classifier": DecisionTreeClassifier(),

    "Random Forest Classifier": RandomForestClassifier()

}
for key, cf in cfs.items():

    cf.fit(X_train, y_train)

    training_score = cross_val_score(cf, X_train, y_train, cv=5)

    print("Classifiers: ", cf.__class__.__name__, 

          "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)



# logistic regression with the best parameters

log_reg = grid_log_reg.best_estimator_



# KNN

knn_params = {"n_neighbors": list(range(2,5,1)), 

              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params)

grid_knn.fit(X_train, y_train)



# KNN with the best parameters

knn = grid_knn.best_estimator_



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, y_train)



# SVC with best parameters

svc = grid_svc.best_estimator_



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, y_train)



# decision tree with best parameters

tree = grid_tree.best_estimator_



# RandomForest Classifier

ran_forest_params = {"criterion": ["gini", "entropy"], "n_estimators": list(range(5,100,5))}

grid_forest = GridSearchCV(RandomForestClassifier(), ran_forest_params)

grid_forest.fit(X_train, y_train)



# decision tree with best parameters

forest = grid_forest.best_estimator_
# implementing the classifiers with their best parameters

print("After determining best parameters:")



log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ',

      round(log_reg_score.mean() * 100, 2).astype(str) + '%')





knears_score = cross_val_score(knn, X_train, y_train, cv=5)

print('Knears Neighbors Cross Validation Score: ',

      round(knears_score.mean() * 100, 2).astype(str) + '%')



svc_score = cross_val_score(svc, X_train, y_train, cv=5)

print('Support Vector Classifier Cross Validation Score: ', 

      round(svc_score.mean() * 100, 2).astype(str) + '%')



tree_score = cross_val_score(tree, X_train, y_train, cv=5)

print('Decision Tree Classifier Cross Validation Score: ',

      round(tree_score.mean() * 100, 2).astype(str) + '%')



forest_score = cross_val_score(forest, X_train, y_train, cv=5)

print('Random Forest Classifier Cross Validation Score: ',

      round(forest_score.mean() * 100, 2).astype(str) + '%')
# use the optimized models to make predictions



log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,

                             method="decision_function")



knn_pred = cross_val_predict(knn, X_train, y_train, cv=5)



svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,

                             method="decision_function")



tree_pred = cross_val_predict(tree, X_train, y_train, cv=5)



forest_pred = cross_val_predict(forest, X_train, y_train, cv=5)
# plot the ROC curves



log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)

knn_fpr, knn_tpr, knn_threshold = roc_curve(y_train, knn_pred)

svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)

tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)

forest_fpr, forest_tpr, forest_threshold = roc_curve(y_train, forest_pred)





def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr,

                             tree_fpr, tree_tpr, forest_fpr, forest_tpr):

    plt.figure(figsize=(16,8))

    plt.title('ROC Curve', fontsize=18)

    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))

    plt.plot(knear_fpr, knear_tpr, label='K Nearest Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knn_pred)))

    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))

    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))

    plt.plot(forest_fpr, forest_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_train, forest_pred)))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.01, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.annotate('Minimum ROC Score of 50%', xy=(0.5, 0.5), xytext=(0.6, 0.3),

                arrowprops=dict(facecolor='#6E726D', shrink=0.05),

                )

    plt.legend()

    

graph_roc_curve_multiple(log_fpr, log_tpr, knn_fpr, knn_tpr, svc_fpr, svc_tpr,

                         tree_fpr, tree_tpr, forest_fpr, forest_tpr)

plt.show()
