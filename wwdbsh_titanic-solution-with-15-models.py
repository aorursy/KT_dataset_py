import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



# machine learning models

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, 

                              BaggingClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



# DNN

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam



# scikit utilities

from sklearn.model_selection import train_test_split
# Load train and test datasets

train_df= pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")

full_data = [train_df, test_df]



train_df.head(3)
col_has_null = {"train_df":[], "test_df":[]}



for idx in range(0, 2):

  df = full_data[idx]

  for col in df:

    if df[col].isnull().values.any():

      if idx == 0: 

        col_has_null["train_df"].append(col)

      else: 

        col_has_null["test_df"].append(col)

        

col_has_null
train_df[["PassengerId", "Survived"]].groupby(["PassengerId"], as_index=False).mean()

train_df.drop(columns=["PassengerId"], inplace=True)
train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)
for df in full_data:

  df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0])

  df.drop(columns=["Name"], inplace=True)

  

#   small_title_group_df = pd.DataFrame(train_df["Title"].value_counts()) # check small title group

#   small_title_group_df.reset_index(inplace=True)

#   small_title_group_df.rename(columns={"index":"Title", "Title":"Count"}, inplace=True)

#   small_title_group_df = small_title_group_df[small_title_group_df["Count"] < 10]

#   small_title_group_df.reset_index(drop=True, inplace=True)

#   print(small_title_group_df)

  

  df["Title"] = df["Title"].replace([" Dr", " Rev", " Col", " Major", " the Countess", " Sir", " Capt", " Don", " Lady", " Jonkheer", " Dona"], " Rare")

  df["Title"] = df["Title"].replace(" Mlle", " Miss")

  df["Title"] = df["Title"].replace(" Ms", " Miss")

  df["Title"] = df["Title"].replace(" Mme", " Mrs")

  df["Title"] = df["Title"].map({" Mr":1, " Mrs":2, " Miss":3, " Master":4, " Rare":5})



train_df[["Title", "Survived"]].groupby(["Title"], as_index=False).mean().sort_values(by="Survived", ascending=False)
for df in full_data:

  df["Sex"] = df["Sex"].map({"female": 1, "male": 0})



train_df[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)
for df in full_data:

  age_mean = df["Age"].mean()

  age_std = df["Age"].std()

  age_null_count = df["Age"].isna().sum()

  

  random_null_list = np.random.randint(age_mean - age_std, age_mean + age_std, size=age_null_count)

  

  df["Age"][np.isnan(df["Age"])] = random_null_list

  df["Age"] = df["Age"].astype(int)

  

  df.loc[df["Age"] <= 16, "Age"] = 0

  df.loc[(df["Age"] > 16) & (df["Age"] <= 32), "Age"] = 1

  df.loc[(df["Age"] > 32) & (df["Age"] <= 48), "Age"] = 2

  df.loc[(df["Age"] > 48) & (df["Age"] <= 64), "Age"] = 3

  df.loc[(df["Age"] > 64) & (df["Age"] <= 80), "Age"] = 4

  

# train_df["AgeGroup"] = pd.cut(train_df["Age"], 5)

train_df[["Age", "Survived"]].groupby(["Age"], as_index=False).mean()
for df in full_data:

  df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

  df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

  df.drop(columns=["SibSp", "Parch", "FamilySize"], inplace=True)



# train_df[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index=False).mean()

train_df[["IsAlone", "Survived"]].groupby(["IsAlone"], as_index=False).mean()
for df in full_data:

  df.drop(columns=["Ticket"], inplace=True)
for df in full_data:

  fare_median = train_df["Fare"].median()

  df["Fare"] = df["Fare"].fillna(fare_median)

  

  df.loc[df["Fare"] <= 7.91, "Fare"] = 0

  df.loc[(df["Fare"] > 7.91) & (df["Fare"] <= 14.454), "Fare"] = 1

  df.loc[(df["Fare"] > 14.454) & (df["Fare"] <= 31.0), "Fare"] = 2

  df.loc[df["Fare"] > 31.0, "Fare"] = 3

  

  df["Fare"] = df["Fare"].astype(int)

  

# train_df["FareClass"] = pd.qcut(train_df["Fare"], 4)

# train_df[["FareClass", "Survived"]].groupby(["FareClass"], as_index=False).mean()

train_df[["Fare", "Survived"]].groupby(["Fare"], as_index=False).mean()
for df in full_data:

  df.drop(columns=["Cabin"], inplace=True)

  

train_df.head()
train_df["Embarked"].value_counts()
train_df["Embarked"] = train_df["Embarked"].fillna("S")



for df in full_data:

  df["Embarked"] = df["Embarked"].map({"S":0, "C":1, "Q":2})

  

train_df[["Embarked", "Survived"]].groupby(["Embarked"], as_index=False).mean()
train_df.isna().sum()
# Pclass

train_df = pd.concat([train_df, pd.get_dummies(train_df["Pclass"], prefix="Pclass")], axis=1)

test_df = pd.concat([test_df, pd.get_dummies(test_df["Pclass"], prefix="Pclass")], axis=1)



# Age

train_df = pd.concat([train_df, pd.get_dummies(train_df["Age"], prefix="Age")], axis=1)

test_df = pd.concat([test_df, pd.get_dummies(test_df["Age"], prefix="Age")], axis=1)



# Fare

train_df = pd.concat([train_df, pd.get_dummies(train_df["Fare"], prefix="Fare")], axis=1)

test_df = pd.concat([test_df, pd.get_dummies(test_df["Fare"], prefix="Fare")], axis=1)



# Embarked

train_df = pd.concat([train_df, pd.get_dummies(train_df["Embarked"], prefix="Embarked")], axis=1)

test_df = pd.concat([test_df, pd.get_dummies(test_df["Embarked"], prefix="Embarked")], axis=1)



# Title

train_df = pd.concat([train_df, pd.get_dummies(train_df["Title"], prefix="Title")], axis=1)

test_df = pd.concat([test_df, pd.get_dummies(test_df["Title"], prefix="Title")], axis=1)
train_df
fig, ax = plt.subplots(figsize=(10, 8))

plt.title("Pearson Correlation of Features", y=1.05, size=15, color="white")

sns.heatmap(train_df.iloc[:,:8].astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap="RdYlGn_r", linecolor="white", annot=True, ax=ax)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
sns.pairplot(train_df.iloc[:,:8], hue="Survived", palette="seismic", size=1., diag_kws=dict(shade=True), plot_kws=dict(s=10))
grid = sns.FacetGrid(train_df, col="Survived", row="Sex", hue="Survived", palette="Set1")

grid.map(plt.hist, "Age", bins=9)
grid = sns.FacetGrid(train_df, col="Survived", row="Pclass", hue="Survived", palette="Set2")

grid.map(plt.hist, "Fare", bins=8)
grid = sns.FacetGrid(train_df, col="Survived", row="IsAlone", hue="Survived", palette="Set1")

grid.map(plt.hist, "Title", bins=9)
grid = sns.FacetGrid(train_df, col="Survived", hue="Survived", palette="Set2")

grid.map(plt.hist, "Embarked", bins=5)
# Delete unnecessary cols

train_df.drop(columns=["Pclass", "Age", "Fare", "Embarked", "Title"], inplace=True)

test_df.drop(columns=["Pclass", "Age", "Fare", "Embarked", "Title"], inplace=True)
x = train_df.iloc[:, 1:].copy()

y = train_df["Survived"].copy()

x_train, x_val, y_train, y_val = train_test_split(x, y)

x_test = test_df.iloc[:, 1:].copy()

# x_train.shape, y_train.shape, x_val, y_val, x_test.shape
# Data frame for comparing

acc_df = pd.DataFrame(columns=["Model", "Acc", "Val_Acc"])
# Logistic Regression

l_reg = LogisticRegression()

l_reg.fit(x_train, y_train)

y_pred = l_reg.predict(x_test)

acc_log = round(l_reg.score(x_train, y_train) * 100, 2)

val_acc_log = round(l_reg.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["LogReg", acc_log, val_acc_log], index=acc_df.columns), ignore_index=True)
# Support Vector Machines

svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

acc_svc = round(svc.score(x_train, y_train) * 100, 2)

val_acc_svc = round(svc.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["SVM", acc_svc, val_acc_svc], index=acc_df.columns), ignore_index=True)
# KNN

knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

acc_knn = round(knn.score(x_train, y_train) * 100, 2)

val_acc_knn = round(knn.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["KNN", acc_knn, val_acc_knn], index=acc_df.columns), ignore_index=True)
# Gaussian Naive Bayes

gnb = GaussianNB()

gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

acc_gnb = round(gnb.score(x_train, y_train) * 100, 2)

val_acc_gnb = round(gnb.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["GaussianNB", acc_gnb, val_acc_gnb], index=acc_df.columns), ignore_index=True)
# Perceptron

perceptron = Perceptron()

perceptron.fit(x_train, y_train)

Y_pred = perceptron.predict(x_test)

acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)

val_acc_perceptron = round(perceptron.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["Perceptron", acc_perceptron, val_acc_perceptron], index=acc_df.columns), ignore_index=True)
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

Y_pred = linear_svc.predict(x_test)

acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

val_acc_linear_svc = round(linear_svc.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["LinearSVC", acc_linear_svc, val_acc_linear_svc], index=acc_df.columns), ignore_index=True)
# Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(x_train, y_train)

Y_pred = sgd.predict(x_test)

acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)

val_acc_sgd = round(sgd.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["SGD", acc_sgd, val_acc_sgd], index=acc_df.columns), ignore_index=True)
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train)

Y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

val_acc_decision_tree = round(decision_tree.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["DecisionTree", acc_decision_tree, val_acc_decision_tree], index=acc_df.columns), ignore_index=True)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)

acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

val_acc_random_forest = round(random_forest.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["RandomForest", acc_random_forest, val_acc_random_forest], index=acc_df.columns), ignore_index=True)
# Voting(lr, knn, dt, rf)

voting = VotingClassifier(

    estimators=[("lr", LogisticRegression(solver="liblinear")),

                ("knn", KNeighborsClassifier()),

                ("dt", DecisionTreeClassifier()),

                ("rf", RandomForestClassifier(n_estimators=100))],

                 voting="soft")

voting.fit(x_train, y_train)

y_pred = voting.predict(x_test)

acc_voting = round(voting.score(x_train, y_train) * 100, 2)

val_acc_voting = round(voting.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["Voting", acc_voting, val_acc_voting], index=acc_df.columns), ignore_index=True)
# Bagging (Decision Tree x 500, random sample count: 100, n_jobs: cpu core count)

bag = BaggingClassifier(

    DecisionTreeClassifier(), n_estimators=500,

    max_samples=100, bootstrap=True, n_jobs=-1

)

bag.fit(x_train, y_train)

y_pred = bag.predict(x_test)

acc_bag = round(bag.score(x_train, y_train) * 100, 2)

val_acc_bag = round(bag.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["Bagging", acc_bag, val_acc_bag], index=acc_df.columns), ignore_index=True)
# AdaBoost

ada = AdaBoostClassifier(

    DecisionTreeClassifier(), n_estimators=500,

    algorithm="SAMME.R", learning_rate=0.01

)

ada.fit(x_train, y_train)

y_pred = ada.predict(x_test)

acc_ada = round(ada.score(x_train, y_train) * 100, 2)

val_acc_ada = round(ada.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["AdaBoost", acc_ada, val_acc_ada], index=acc_df.columns), ignore_index=True)
# Gradient Boosting Classifier

gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01)

gbc.fit(x_train, y_train)

y_pred = gbc.predict(x_test)

acc_gbc = round(gbc.score(x_train, y_train) * 100, 2)

val_acc_gbc = round(gbc.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["GBC", acc_gbc, val_acc_gbc], index=acc_df.columns), ignore_index=True)
# XGBoosting Classifier

xgb = XGBClassifier(n_estimators=500, learning_rate=0.01)

xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)

acc_xgb = round(xgb.score(x_train, y_train) * 100, 2)

val_acc_xgb = round(xgb.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["XGB", acc_xgb, val_acc_xgb], index=acc_df.columns), ignore_index=True)
# LightGBM Classifier (faster working speed than xgboost but has a risk for overfitting)

lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.01, max_depth=2)

lgbm.fit(x_train, y_train)

y_pred = lgbm.predict(x_test)

acc_lgbm = round(lgbm.score(x_train, y_train) * 100, 2)

val_acc_lgbm = round(lgbm.score(x_val, y_val) * 100, 2)

acc_df = acc_df.append(pd.Series(["LGBM", acc_lgbm, val_acc_lgbm], index=acc_df.columns), ignore_index=True)
# DNN

epochs = 2000



model = Sequential()

model.add(Dense(512, input_shape=(x_train.shape[1],), kernel_initializer="he_uniform", activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))

model.add(Dropout(0.3))

model.add(Dense(128, kernel_initializer="he_uniform", activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))

model.add(Dropout(0.3))

model.add(Dense(128, kernel_initializer="he_uniform", activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))

model.add(Dense(1, activation="sigmoid"))



model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])

hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=16, verbose=1)

y_pred = (model.predict(x_test).flatten() > 0.5).astype(int)

acc_df = acc_df.append(pd.Series(["DNN", round(np.max(hist.history["accuracy"]) * 100, 2), round(np.max(hist.history["val_accuracy"]) * 100, 2)], index=acc_df.columns), ignore_index=True)
# summarize history for accuracy

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
acc_df["Variance"] = np.abs(acc_df["Acc"] - acc_df["Val_Acc"])
acc_df.sort_values(by="Acc", ascending=False).reset_index(drop=True)
acc_df.sort_values(by="Variance", ascending=True).reset_index(drop=True)
submission_df = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": y_pred})

submission_df.to_csv("submission.csv", index=False)