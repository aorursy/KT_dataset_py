# Basic libraries

import numpy as np

import pandas as pd

import warnings

warnings.simplefilter('ignore')



# Directry check

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Data preprocessing

import datetime

import re

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# Visualization

from matplotlib import pyplot as plt

import folium

import seaborn as sns



# Over sampling method, SMOTE

from imblearn.over_sampling import SMOTE



# Logistic regression

from sklearn.linear_model import LogisticRegression



# KNeithborsClassfier

from sklearn.neighbors import KNeighborsClassifier



# parameter opimization

from sklearn.model_selection import GridSearchCV



# Multilayer perceptron

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

from keras.callbacks import ModelCheckpoint, EarlyStopping



# Validation

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc
sample = pd.read_csv("/kaggle/input/GiveMeSomeCredit/sampleEntry.csv")

df_train = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv", header=0)

data_dict = pd.read_excel("/kaggle/input/GiveMeSomeCredit/Data Dictionary.xls")

df_test = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-test.csv", header=0)
sample.head()
data_dict
df_train.head()
df_test.head()
# Rename columns

colnames=["Id", "Probability", "RUUL", "age", "Time_30", "D_ratio", "M_income", "Ooen_loan", "Times_90", "E_loan", "Time_60", "Dependents"]



df_train.columns=colnames

df_test.columns=colnames
# data size

print("train data size:{}".format(df_train.shape))

print("test data size:{}".format(df_test.shape))
# data info

print("train data info:\n{}".format(df_train.info()))

print("-"*50)

print("test data info:\n{}".format(df_test.info()))
# Null data check

print("train data")

print(df_train.isnull().sum())

print("-"*50)

print("test data")

print(df_test.isnull().sum())
with plt.style.context("fivethirtyeight"):

    sns.countplot(df_train["Probability"])

    plt.title("Traget value count plot\n(0:negative, 1:positive)")
negative_df = df_train.query("Probability==0")

positive_df = df_train.query("Probability==1")



with plt.style.context("fivethirtyeight"):

    fig, ax = plt.subplots(2,5,figsize=(30,12))

    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i in range(0,5):

        for j in range(0,5):

            sns.distplot(negative_df.iloc[:,2+i], ax=ax[0,i], kde=False, color="blue", label="negative")

            sns.distplot(positive_df.iloc[:,2+i], ax=ax[0,i], kde=False, color="red", label="positive")

            ax[0,i].set_title("{}".format(negative_df.columns[2+i]))

            ax[0,i].set_yscale("log")

            ax[0,i].legend(labels=["negetive", "positive"])

            sns.distplot(negative_df.iloc[:,7+j], ax=ax[1,j], kde=False, color="blue", label="negative")

            sns.distplot(positive_df.iloc[:,7+j], ax=ax[1,j], kde=False, color="red", label="positive")

            ax[1,j].set_title("{}".format(negative_df.columns[7+j]))

            ax[1,j].set_yscale("log")

            ax[1,j].legend(labels=["negetive", "positive"])
# plot

with plt.style.context("fivethirtyeight"):

    sns.pairplot(df_train.sample(5000).iloc[:, 2:])
# correlation

matrix = df_train.sample(5000).iloc[:, 2:].fillna(0) # tempolary fill na =0

cols = df_train.sample(5000).iloc[:, 2:].columns

cm = np.corrcoef(matrix.T)



with plt.style.context("fivethirtyeight"):

    sns.set(font_scale=1.0)

    plt.figure(figsize=(10,10))

    hm = sns.heatmap(cm,

                cbar=True,

                annot=True,

                square=True,

                cmap="RdBu_r",

                fmt=".2f",

                annot_kws={"size":10},

                yticklabels=cols,

                xticklabels=cols,

                vmax=1,

                vmin=-1,

                center=0)
medi_m_income = df_train["M_income"].median()

medi_dependents = df_train["Dependents"].median()



# M_income, train_data and test_data

df_train["M_income"].fillna(medi_m_income, inplace=True)

df_train["M_income"].fillna(medi_m_income, inplace=True)

# Dependents, train_data and test_data

df_train["Dependents"].fillna(medi_dependents, inplace=True)

df_train["Dependents"].fillna(medi_dependents, inplace=True)
df_train_ml = df_train.copy()



# age 0 is dropped

df_train_ml = df_train_ml.query("age != 0")



# Other data preprocessing, over quantile99.99% is dropped.

RUUL_99 = df_train_ml["RUUL"].quantile(0.9999)

D_ratio_99 = df_train_ml["D_ratio"].quantile(0.9999)

M_income_99 = df_train_ml["M_income"].quantile(0.9999)

E_loan_99 = df_train_ml["E_loan"].quantile(0.9999)

Dependents_99 = df_train_ml["Dependents"].quantile(0.9999)



df_train_ml = df_train_ml[df_train_ml["RUUL"]<RUUL_99]

df_train_ml = df_train_ml[df_train_ml["D_ratio"]<D_ratio_99]

df_train_ml = df_train_ml[df_train_ml["M_income"]<M_income_99]

df_train_ml = df_train_ml[df_train_ml["E_loan"]<E_loan_99]

df_train_ml = df_train_ml[df_train_ml["Dependents"]<Dependents_99]



df_train_ml = df_train_ml
negative_df = df_train_ml.query("Probability==0")

positive_df = df_train_ml.query("Probability==1")



with plt.style.context("fivethirtyeight"):

    fig, ax = plt.subplots(2,5,figsize=(30,12))

    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i in range(0,5):

        for j in range(0,5):

            sns.distplot(negative_df.iloc[:,2+i], ax=ax[0,i], kde=False, color="blue", label="negative")

            sns.distplot(positive_df.iloc[:,2+i], ax=ax[0,i], kde=False, color="red", label="positive")

            ax[0,i].set_title("{}".format(negative_df.columns[2+i]))

            ax[0,i].set_yscale("log")

            ax[0,i].legend(labels=["negetive", "positive"])

            sns.distplot(negative_df.iloc[:,7+j], ax=ax[1,j], kde=False, color="blue", label="negative")

            sns.distplot(positive_df.iloc[:,7+j], ax=ax[1,j], kde=False, color="red", label="positive")

            ax[1,j].set_title("{}".format(negative_df.columns[7+j]))

            ax[1,j].set_yscale("log")

            ax[1,j].legend(labels=["negetive", "positive"])
# Data

X = df_train_ml[['RUUL', 'age', 'Time_30', 'D_ratio', 'M_income','Ooen_loan', 'Times_90', 'E_loan', 'Time_60', 'Dependents']]

y = df_train_ml['Probability']



# Data standarlization

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



# Create StandardScaler instance and fit_trainsform

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.fit_transform(X_test)



# Create SMOTE instance

smote = SMOTE(sampling_strategy="auto", random_state=10)



# data split

X_train_resampled, y_train_resampled = smote.fit_sample(X_train_std, y_train)
# Create logistic regression instance

lr = LogisticRegression()



# Grid search

param_range = [0.001, 0.01, 0.1, 1.0, 10, 100]

penalty = ['l1', 'l2']

param_grid = [{"C":param_range, "penalty":penalty}]



gs = GridSearchCV(estimator=lr, param_grid=param_grid, scoring="recall", cv=10, n_jobs=-1)

gs = gs.fit(X_train_resampled, y_train_resampled)



print(gs.best_score_.round(3))

print(gs.best_params_)
clf_lr = gs.best_estimator_

print('Test accuracy: %.3f' % clf_lr.score(X_test_std, y_test))
y_pred = clf_lr.predict(X_test_std)

y_pred_train = clf_lr.predict(X_train_std)



# Validation of model

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))



print("*accuracy_train = %.3f" % accuracy_score(y_true=y_train, y_pred=y_pred_train))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))



print("*precision_train = %.3f" % precision_score(y_true=y_train, y_pred=y_pred_train))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))



print("*recall_train = %.3f" % recall_score(y_true=y_train, y_pred=y_pred_train))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))



print("*f1_score_train = %.3f" % f1_score(y_true=y_train, y_pred=y_pred_train))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))
# ROC curve and AUC

y_score = clf_lr.predict_proba(X_test_std)[:, 1]



fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)

# Visualization

with plt.style.context("fivethirtyeight"):

    plt.figure(figsize=(10,6))

    plt.plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr))

    plt.plot([0,1], [0,1], linestyle='--', label='random')

    plt.plot([0,0,1], [0,1,1], linestyle='--', label="ideal")

    plt.legend()

    plt.xlabel("false positive rate")

    plt.ylabel("true positive rate")
# KNeithborsClassfier

knn = KNeighborsClassifier(metric='minkowski')



# Grid search

param_range = [5, 10, 15, 20]

param_grid = [{"n_neighbors":param_range, "p":[1,2]}]



gs_knn = GridSearchCV(estimator=knn, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_knn = gs_knn.fit(X_train_std, y_train)



print(gs_knn.best_score_.round(3))

print(gs_knn.best_params_)
clf_knn = gs.best_estimator_

print('Test accuracy: %.3f' % clf_knn.score(X_test_std, y_test))
y_pred = clf_knn.predict(X_test_std)

y_pred_train = clf_knn.predict(X_train_std)



# Validation of model

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))



print("*accuracy_train = %.3f" % accuracy_score(y_true=y_train, y_pred=y_pred_train))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))



print("*precision_train = %.3f" % precision_score(y_true=y_train, y_pred=y_pred_train))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))



print("*recall_train = %.3f" % recall_score(y_true=y_train, y_pred=y_pred_train))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))



print("*f1_score_train = %.3f" % f1_score(y_true=y_train, y_pred=y_pred_train))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))
# ROC curve and AUC

y_score = clf_knn.predict_proba(X_test_std)[:, 1]



fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)

# Visualization

with plt.style.context("fivethirtyeight"):

    plt.figure(figsize=(10,6))

    plt.plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr))

    plt.plot([0,1], [0,1], linestyle='--', label='random')

    plt.plot([0,0,1], [0,1,1], linestyle='--', label="ideal")

    plt.legend()

    plt.xlabel("false positive rate")

    plt.ylabel("true positive rate")
y_train_resampled = np.expand_dims(y_train_resampled, axis=1)

y_test = np.expand_dims(y_test, axis=1)



# shape check

print("X_train_resampled: %s, y_train_resampled: %s" % (X_train_resampled.shape, y_train_resampled.shape))

print("X_test: %s, y_test: %s" % (X_test.shape, y_test.shape))
# Model

model = Sequential()

model.add(Dense(8, activation='relu', input_dim=10))

model.add(Dense(5, activation='relu'))

model.add(Dense(3, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer=SGD(lr=0.01), loss='mse')
mc = ModelCheckpoint("model_01.h2", monitor="val_loss", save_best_only=True, verbose=1)

es = EarlyStopping(monitor='val_loss', patience=10)



hist = model.fit(X_train_resampled, y_train_resampled,

                 callbacks=[mc, es],

                 epochs=100, batch_size=16,

                 validation_split=0.2, verbose=2)
# visualization loss plot

train_loss = hist.history["loss"]

val_loss = hist.history["val_loss"]

with plt.style.context("fivethirtyeight"):

    plt.figure(figsize=(8, 4))

    plt.plot(range(len(train_loss)), train_loss, label='train_loss')

    plt.plot(range(len(val_loss)), val_loss, label='valid_loss')

    plt.xlabel('epoch', fontsize=16)

    plt.ylabel('loss', fontsize=16)

    plt.yscale('log')

    plt.legend(fontsize=16)
# test data prediction

y_pred = model.predict(X_test_std)

y_pred = [num[0] for num in y_pred]

y_pred_df = pd.DataFrame({"y_pred":y_pred})



# train data prediction

y_pred_train = model.predict(X_train_std)

y_pred_train = [num[0] for num in y_pred_train]

y_pred_train_df = pd.DataFrame({"y_pred":y_pred_train})
# y_pred flg

def prediction_flg(x):

    if x["y_pred"] > 0.5:

        res=1

    else:

        res=0

    return res



y_pred_df["y_flg"] = y_pred_df.apply(prediction_flg, axis=1)

y_pred_train_df["y_flg"] = y_pred_train_df.apply(prediction_flg, axis=1)
y_pred = y_pred_df["y_flg"]

y_pred_train = y_pred_train_df["y_flg"]



# Validation of model

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))



print("*accuracy_train = %.3f" % accuracy_score(y_true=y_train, y_pred=y_pred_train))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))



print("*precision_train = %.3f" % precision_score(y_true=y_train, y_pred=y_pred_train))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))



print("*recall_train = %.3f" % recall_score(y_true=y_train, y_pred=y_pred_train))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))



print("*f1_score_train = %.3f" % f1_score(y_true=y_train, y_pred=y_pred_train))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))
# ROC curve and AUC

y_score = model.predict_proba(X_test_std)[:, 0]



fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)

# Visualization

with plt.style.context("fivethirtyeight"):

    plt.figure(figsize=(10,6))

    plt.plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr))

    plt.plot([0,1], [0,1], linestyle='--', label='random')

    plt.plot([0,0,1], [0,1,1], linestyle='--', label="ideal")

    plt.legend()

    plt.xlabel("false positive rate")

    plt.ylabel("true positive rate")
X_Test = df_test[['RUUL', 'age', 'Time_30', 'D_ratio', 'M_income','Ooen_loan', 'Times_90', 'E_loan', 'Time_60', 'Dependents']]



X_Test_std = sc.fit_transform(X_Test)



# submit Test data prediction

y_Pred = model.predict(X_Test_std)

y_Pred = [num[0] for num in y_Pred]

y_Pred_df = pd.DataFrame({"y_pred":y_Pred})
submit = pd.DataFrame({"Id":df_test["Id"],

                     "Probability":y_Pred_df["y_pred"]})
submit.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")