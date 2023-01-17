# Import basic libraries

import numpy as np

import pandas as pd



# Graph visualization library

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('fivethirtyeight')



# Data preprocessing library

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# Machine learning library

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

import lightgbm as lgb



# Evaluation library

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc
df_train = pd.read_csv("../input/titanic/train.csv", header=0)

df_train.head()
df_test = pd.read_csv("../input/titanic/test.csv", header=0)

df_test.head()
# Data size

print("Training data size:{}".format(df_train.shape))

print("Test data size:{}".format(df_test.shape))
# Data info, training_data

df_train.info()
# Data info, test_data

df_test.info()
# Null data check, training_data

df_train.isnull().sum()
# Null data check, test_data

df_test.isnull().sum()
# Unique count check, about string data.

# Ticket data

df_train["Ticket"].value_counts()
# Cabin data

df_train["Cabin"].value_counts()
print("0 count:{}".format(df_train.query("Survived==0").shape[0]))

print("1 count:{}".format(df_train.query("Survived==1").shape[0]))

sns.countplot(df_train["Survived"])

plt.title("Survived flag count for training data\n 0 for deceased, 1 for survived")
fig, ax = plt.subplots(1,5, figsize=(25,4))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# Pclass

sns.countplot(df_train["Pclass"], ax=ax[0])

ax[0].set_title("Passenger class")



# Sex

sns.countplot(df_train["Sex"], ax=ax[1])

ax[1].set_title("Sex")



# SibSp

sns.countplot(df_train["SibSp"], ax=ax[2])

ax[2].set_title("family relations SibSp")



# Parch

sns.countplot(df_train["Parch"], ax=ax[3])

ax[3].set_title("family relations Parch")



# Embarked

sns.countplot(df_train["Embarked"], ax=ax[4])

ax[4].set_title("Embarked")
# Visualization of ratio with barplot

fig, ax = plt.subplots(1,5, figsize=(25,4))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# Pclass

# Pivot_table

pivot = pd.pivot_table(df_train, index="Pclass", columns="Survived", values="PassengerId", aggfunc="count").reset_index()

pivot["sum"] = pivot[0] + pivot[1]



ax[0].bar(pivot["Pclass"], pivot[0]/pivot["sum"]*100, color='blue', alpha=0.5)

ax[0].bar(pivot["Pclass"], pivot[1]/pivot["sum"]*100, bottom=pivot[0]/pivot["sum"]*100, color='red', alpha=0.5)

ax[0].set_xlim([0.5,3.5])

ax[0].set_xticks([1,2,3])

ax[0].set_title("Passenger class")

ax[0].legend(labels=["deceased", "survived"], loc="lower right", facecolor="white")



# Sex

# Pivot_table

pivot = pd.pivot_table(df_train, index="Sex", columns="Survived", values="PassengerId", aggfunc="count").reset_index()

pivot["sum"] = pivot[0] + pivot[1]



ax[1].bar(pivot["Sex"], pivot[0]/pivot["sum"]*100, color='blue', alpha=0.5)

ax[1].bar(pivot["Sex"], pivot[1]/pivot["sum"]*100, bottom=pivot[0]/pivot["sum"]*100, color='red', alpha=0.5)

ax[1].set_title("Sex")

ax[1].legend(labels=["deceased", "survived"], loc="lower right", facecolor="white")



# SibSp

# Pivot_table

pivot = pd.pivot_table(df_train, index="SibSp", columns="Survived", values="PassengerId", aggfunc="count").reset_index()

pivot["sum"] = pivot[0] + pivot[1]



ax[2].bar(pivot["SibSp"], pivot[0]/pivot["sum"]*100, color='blue', alpha=0.5)

ax[2].bar(pivot["SibSp"], pivot[1]/pivot["sum"]*100, bottom=pivot[0]/pivot["sum"]*100, color='red', alpha=0.5)

ax[2].set_xlim([-0.5,2.5])

ax[2].set_xticks([0,1,2])

ax[2].set_title("SibSp")

ax[2].legend(labels=["deceased", "survived"], loc="lower right", facecolor="white")



# Parch

# Pivot_table

pivot = pd.pivot_table(df_train, index="Parch", columns="Survived", values="PassengerId", aggfunc="count").reset_index()

pivot["sum"] = pivot[0] + pivot[1]



ax[3].bar(pivot["Parch"], pivot[0]/pivot["sum"]*100, color='blue', alpha=0.5)

ax[3].bar(pivot["Parch"], pivot[1]/pivot["sum"]*100, bottom=pivot[0]/pivot["sum"]*100, color='red', alpha=0.5)

ax[3].set_xlim([-0.5,2.5])

ax[3].set_xticks([0,1,2])

ax[3].set_title("Parch")

ax[3].legend(labels=["deceased", "survived"], loc="lower right", facecolor="white")



# Embarked

# Pivot_table

pivot = pd.pivot_table(df_train, index="Embarked", columns="Survived", values="PassengerId", aggfunc="count").reset_index()

pivot["sum"] = pivot[0] + pivot[1]



ax[4].bar(pivot["Embarked"], pivot[0]/pivot["sum"]*100, color='blue', alpha=0.5)

ax[4].bar(pivot["Embarked"], pivot[1]/pivot["sum"]*100, bottom=pivot[0]/pivot["sum"]*100, color='red', alpha=0.5)

ax[4].set_title("Embarked")

ax[4].legend(labels=["deceased", "survived"], loc="lower right", facecolor="white")
fig, ax = plt.subplots(1,2, figsize=(10,4))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# Age *Null data is dropped, tempolary. 

sns.distplot(df_train.dropna()["Age"], ax=ax[0], kde=False, bins=20)

ax[0].set_title("Age")

ax[0].set_ylabel("count")



# Fare

sns.distplot(df_train["Fare"], ax=ax[1], kde=False, bins=20)

ax[1].set_title("Fare")

ax[1].set_ylabel("count")
fig, ax = plt.subplots(1,2, figsize=(10,4))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# Age *Null data is dropped, tempolary. 

sns.distplot(df_train.dropna().query("Survived==0")["Age"], ax=ax[0], kde=False, bins=20, norm_hist=True)

sns.distplot(df_train.dropna().query("Survived==1")["Age"], ax=ax[0], kde=False, bins=20, norm_hist=True)

ax[0].set_title("Age")

ax[0].set_ylabel("count")

ax[0].legend(labels=["deceased", "survived"], loc="upper right", facecolor="white")



# Fare

sns.distplot(df_train.query("Survived==0")["Fare"], ax=ax[1], kde=False, bins=20, norm_hist=True)

sns.distplot(df_train.query("Survived==1")["Fare"], ax=ax[1], kde=False, bins=20, norm_hist=True)

ax[1].set_title("Fare")

ax[1].set_ylabel("count")

ax[1].legend(labels=["deceased", "survived"], loc="upper right", facecolor="white")
# training data

df_train["Age"].fillna(np.mean(df_train["Age"]), inplace=True)

df_train["Fare"].fillna(np.mean(df_train["Fare"]), inplace=True)

df_train["Embarked"].fillna('S', inplace=True)

df_train.dropna(axis=1, inplace=True)



# test data

df_test["Age"].fillna(np.mean(df_train["Age"]), inplace=True)

df_test["Fare"].fillna(np.mean(df_train["Fare"]), inplace=True)

df_test.dropna(axis=1, inplace=True)
# Sex

def sex(x):

    if x["Sex"] == "male":

        res = 0

    else :

        res = 1

    return res



df_train["Sex_cate"] = df_train.apply(sex, axis=1)

df_test["Sex_cate"] = df_test.apply(sex, axis=1)



# Age

def age_band(x):

    if x["Age"] <= 10:

        res = 0

    elif x["Age"] <= 20 and x["Age"] > 10:

        res = 1

    elif x["Age"] <= 30 and x["Age"] > 20:

        res = 2

    elif x["Age"] <= 40 and x["Age"] > 30:

        res = 3

    elif x["Age"] <= 50 and x["Age"] > 40:

        res = 4

    elif x["Age"] <= 60 and x["Age"] > 50:

        res = 5

    else :

        res = 6

    return res



df_train["Age_band"] = df_train.apply(age_band, axis=1)

df_test["Age_band"] = df_test.apply(age_band, axis=1)



# Fare

def fare_band(x):

    if x["Fare"] <= 25:

        res = 0

    elif x["Fare"] <= 50 and x["Fare"] > 25:

        res = 1

    elif x["Fare"] <= 75 and x["Fare"] > 50:

        res = 2

    elif x["Fare"] <= 100 and x["Fare"] > 75:

        res = 3

    elif x["Fare"] <= 125 and x["Fare"] > 100:

        res = 4

    else :

        res = 5

    return res



df_train["Fare_band"] = df_train.apply(fare_band, axis=1)

df_test["Fare_band"] = df_test.apply(fare_band, axis=1)



# Embarked

def embarked_flg(x):

    if x["Embarked"] == 'S':

        res = 0

    elif x["Embarked"] == 'C':

        res = 1

    else:

        res = 2

    return res



df_train["Embarked_flg"] = df_train.apply(embarked_flg, axis=1)

df_test["Embarked_flg"] = df_test.apply(embarked_flg, axis=1)
# Confirming the dataframe.

df_train.head()
# Checking by visualization with heatmap.

plt.figure(figsize=(12,8))

hm = sns.heatmap(df_train[['Survived', 'Pclass','SibSp','Parch','Sex_cate', 'Age_band', 'Fare_band', 'Embarked_flg']].corr(),

                cbar=True,

                annot=True,

                square=True,

                cmap="RdBu_r",

                fmt=".2f",

                annot_kws={"size":10},

                yticklabels=df_train[['Survived', 'Pclass','SibSp','Parch','Sex_cate', 'Age_band', 'Fare_band', 'Embarked_flg']].columns,

                vmax=1,

                vmin=-1,

                center=0)

plt.xlabel("Variables")

plt.ylabel("Variables")
# Make the target data and explanatry data

X = df_train[['Pclass','SibSp','Parch','Sex_cate', 'Age_band', 'Fare_band', 'Embarked_flg']]

y = df_train[['Survived']]



# Data splitting to make the training data and validation data

# training data :80%, validation(test data) :20%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# Taking veryfing to Standarlized data

sc = StandardScaler()

sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
# Logistic Regression

lr = LogisticRegression()



param_range = [0.001, 0.01, 0.1, 1.0]

penalty = ['l1', 'l2']

param_grid = [{"C":param_range, "penalty":penalty}]



gs_lr = GridSearchCV(estimator=lr, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_lr = gs_lr.fit(X_train_std, y_train)



print(gs_lr.best_score_.round(3))

print(gs_lr.best_params_)
# Support vector machine, SVM

svm = SVC(random_state=10, probability=True)



param_range = [0.001, 0.01, 0.1, 1.0]

param_grid = [{'C':param_range, 'kernel':['linear']}]



gs_svm = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

gs_svm = gs_svm.fit(X_train_std, y_train)



print(gs_svm.best_score_.round(3))

print(gs_svm.best_params_)
# KNeithborsClassfier

knn = KNeighborsClassifier(metric='minkowski')



param_range = [5, 10, 15, 20]

param_grid = [{"n_neighbors":param_range, "p":[1,2]}]



gs_knn = GridSearchCV(estimator=knn, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_knn = gs_knn.fit(X_train_std, y_train)



print(gs_knn.best_score_.round(3))

print(gs_knn.best_params_)
# Decision tree

tree = DecisionTreeClassifier(max_depth=4, random_state=10)



param_range = [3, 6, 9, 12]

leaf = [10, 15, 20]

criterion = ["entropy", "gini", "error"]

param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



gs_tree = GridSearchCV(estimator=tree, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_tree = gs_tree.fit(X_train, y_train)



print(gs_tree.best_score_.round(3))

print(gs_tree.best_params_)
# Random Forest

forest = RandomForestClassifier(n_estimators=100, random_state=10)



param_range = [5, 10, 15]

criterion = ["entropy", "gini", "error"]

param_grid = [{"n_estimators":param_range, "criterion":criterion}]



gs_forest = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_forest = gs_forest.fit(X_train, y_train)



print(gs_forest.best_score_.round(3))

print(gs_forest.best_params_)
# XGB

xgbc = xgb.XGBClassifier(random_state=10)



# prameters

max_depth = [10, 15, 20, 25]

min_samples_leaf = [1,3,5]

min_samples_split = [1,2,4]



param_grid = [{"max_depth":max_depth,

               "min_samples_leaf":min_samples_leaf, "min_samples_split":min_samples_split}]



# Optimization by Grid search

gs_xgb = GridSearchCV(estimator=xgbc, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)



gs_xgb = gs_xgb.fit(X_train, y_train)



print(gs_xgb.best_score_)

print(gs_xgb.best_params_)
# LGBM

lgbm = lgb.LGBMClassifier()



# prameters

max_depth = [3, 5, 10]

min_samples_leaf = [1,3,5,7]

min_samples_split = [2, 4, 6, 8]



param_grid = [{"max_depth":max_depth,

               "min_samples_leaf":min_samples_leaf, "min_samples_split":min_samples_split}]



# Optimization by Grid search

gs_lgbm = GridSearchCV(estimator=lgbm, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)

gs_lgbm = gs_lgbm.fit(X_train, y_train)



print(gs_lgbm.best_score_.round(3))

print(gs_lgbm.best_params_)
print("-"*50)

# Logistic Regression Result

y_pred = gs_lr.best_estimator_.predict(X_test_std)

print("Logistic Regression Result")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)



# Support vector machine, SVM

y_pred = gs_svm.best_estimator_.predict(X_test_std)

print("Support vector machine, SVM")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)



# KNeithborsClassfier

y_pred = gs_knn.best_estimator_.predict(X_test_std)

print("KNeithborsClassfier")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)



# Decision tree

y_pred = gs_tree.best_estimator_.predict(X_test)

print("Decision tree")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)



# Random Forest

y_pred = gs_forest.best_estimator_.predict(X_test)

print("Random Forest")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)



# XGB

y_pred = gs_xgb.best_estimator_.predict(X_test)

print("XGB")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)



# LGBM

y_pred = gs_lgbm.best_estimator_.predict(X_test)

print("LGBM")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)
# Visualization of roc curve

fig, ax = plt.subplots(2,4, figsize=(25, 10))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# Logistic Regression

y_score = gs_lr.best_estimator_.predict_proba(X_test_std)[:, 1]

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)



ax[0,0].plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr))

ax[0,0].plot([0,1], [0,1], linestyle='--', label='random')

ax[0,0].plot([0,0,1], [0,1,1], linestyle='--', label="ideal")

ax[0,0].legend()

ax[0,0].set_xlabel("false positive rate")

ax[0,0].set_ylabel("true positive rate")

ax[0,0].set_title("Logistic Regression")



# Support vector machine, SVM

y_score = gs_svm.best_estimator_.predict_proba(X_test_std)[:, 1]

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)



ax[0,1].plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr))

ax[0,1].plot([0,1], [0,1], linestyle='--', label='random')

ax[0,1].plot([0,0,1], [0,1,1], linestyle='--', label="ideal")

ax[0,1].legend()

ax[0,1].set_xlabel("false positive rate")

ax[0,1].set_ylabel("true positive rate")

ax[0,1].set_title("Support vector machine, SVM")



# KNeithborsClassfier

y_score = gs_knn.best_estimator_.predict_proba(X_test_std)[:, 1]

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)



ax[0,2].plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr))

ax[0,2].plot([0,1], [0,1], linestyle='--', label='random')

ax[0,2].plot([0,0,1], [0,1,1], linestyle='--', label="ideal")

ax[0,2].legend()

ax[0,2].set_xlabel("false positive rate")

ax[0,2].set_ylabel("true positive rate")

ax[0,2].set_title("KNeithborsClassfier")



# Decision tree

y_score = gs_tree.best_estimator_.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)



ax[0,3].plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr))

ax[0,3].plot([0,1], [0,1], linestyle='--', label='random')

ax[0,3].plot([0,0,1], [0,1,1], linestyle='--', label="ideal")

ax[0,3].legend()

ax[0,3].set_xlabel("false positive rate")

ax[0,3].set_ylabel("true positive rate")

ax[0,3].set_title("Decision tree")



# Forest

y_score = gs_forest.best_estimator_.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)



ax[1,0].plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr))

ax[1,0].plot([0,1], [0,1], linestyle='--', label='random')

ax[1,0].plot([0,0,1], [0,1,1], linestyle='--', label="ideal")

ax[1,0].legend()

ax[1,0].set_xlabel("false positive rate")

ax[1,0].set_ylabel("true positive rate")

ax[1,0].set_title("Random Forest")



# xgb

y_score = gs_xgb.best_estimator_.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)



ax[1,1].plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr))

ax[1,1].plot([0,1], [0,1], linestyle='--', label='random')

ax[1,1].plot([0,0,1], [0,1,1], linestyle='--', label="ideal")

ax[1,1].legend()

ax[1,1].set_xlabel("false positive rate")

ax[1,1].set_ylabel("true positive rate")

ax[1,1].set_title("XGB")



# lgbm

y_score = gs_lgbm.best_estimator_.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)



ax[1,2].plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr))

ax[1,2].plot([0,1], [0,1], linestyle='--', label='random')

ax[1,2].plot([0,0,1], [0,1,1], linestyle='--', label="ideal")

ax[1,2].legend()

ax[1,2].set_xlabel("false positive rate")

ax[1,2].set_ylabel("true positive rate")

ax[1,2].set_title("LGBM")

test = df_test[['Pclass', 'SibSp', 'Parch', 'Sex_cate', 'Age_band', 'Fare_band', 'Embarked_flg']]



# Decision tree

y_pred_test = gs_tree.best_estimator_.predict(test)



submit = pd.DataFrame({"PassengerId":df_test["PassengerId"], "Survived":y_pred_test})
submit.head()
submit.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
# prediction

y_pred_test_lgbm = gs_lgbm.best_estimator_.predict(test)

y_pred_test_xgb = gs_xgb.best_estimator_.predict(test)



# Ensemble

y_pred_test_en = (y_pred_test*0.3 + y_pred_test_lgbm*0.4 + y_pred_test_xgb*0.3)



submit_en = pd.DataFrame({"PassengerId":df_test["PassengerId"], "Survived":y_pred_test_en}).round(0)

submit_en["Survived"] = [int(i) for i in submit_en["Survived"]]
submit_en.to_csv('my_submission_en.csv', index=False)

print("Your submission was successfully saved!")
submit_en
submit