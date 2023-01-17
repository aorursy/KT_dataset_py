import numpy as np 

import pandas as pd 

import os

import itertools

import seaborn as sns

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
Train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

Train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")
Train = Train_transaction.merge(Train_identity, on='TransactionID', how="left")

Train = Train.drop(Train[Train.isFraud == 0].iloc[:300000].index)

del Train_transaction, Train_identity
Test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")

Test_identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")
Test = Test_transaction.merge(Test_identity, on='TransactionID', how="left")

del Test_transaction, Test_identity
def Change(x):

    for col in x.select_dtypes(include=['object']).columns:

               x[col] = x[col].astype('category')

    for col in x.select_dtypes(include=['category']).columns: 

               x[col] = x[col].cat.codes

    return x  
Train = Change(Train)

Test = Change(Test)
sns.countplot(Train.isFraud);
Correalation_Matrix = Train[Train.columns[1:]].corr()['isFraud'][:]

Correlation_Matrix = pd.DataFrame(Correalation_Matrix)
Correlation_Matrix.isFraud.plot(figsize=(15,7))

plt.ylabel('Correlation Score')

plt.xlabel('Features')
Column = abs(Correalation_Matrix)

Column = pd.DataFrame(Column)

Column = Column.reset_index()

Column.rename(columns={'index':'Features','isFraud':'Correlation_score'}, inplace=True)

Column.head(5)
relevant = []

for i in range(len(Column)):

    if Column.Correlation_score[i] > 0.30:

           relevant.append(Column.Features[i])

    else:

        continue
def keep_cols(DataFrame, keep_these):

    drop_these = list(set(list(DataFrame)) - set(keep_these))

    return DataFrame.drop(drop_these, axis = 1)



DF = Train.pipe(keep_cols, relevant)

DF.head()
DF.describe().head(5)
plt.figure(figsize=(20,15))

cor = DF.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
DF.corrwith(DF.isFraud).plot.bar(

        figsize = (20, 10), title = "Correlation with class", fontsize = 15,

        rot = 45, grid = True)
y = Train.isFraud

Train.drop("isFraud", inplace=True, axis=1)

Train.head(5)
Train.describe()
Train = Train.fillna(-999)

Test = Test.fillna(-999)
X_train, X_test, y_train, y_test = train_test_split(Train, y, test_size=0.3, stratify = y, random_state=3)
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                              colsample_bynode=1, colsample_bytree=0.9, gamma=0,

                              learning_rate=0.4, max_delta_step=0, max_depth=14,

                              min_child_weight=1, missing=-999, n_estimators=600, n_jobs=-1,

                              nthread=None, objective='binary:logistic', random_state=0,

                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

                              silent=None, subsample=0.9, verbosity=0, tree_method='gpu_hist')
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

metrics.accuracy_score(y_test, y_pred)*100
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



y_pred = model.predict(X_test)

confusion_mtx = confusion_matrix(y_test, y_pred) 

plot_confusion_matrix(confusion_mtx, classes = range(2)) 
feature_importance = model.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = sorted_idx[len(feature_importance) - 50:]

pos = np.arange(sorted_idx.shape[0]) + .5



plt.figure(figsize=(10,12))

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
probablities = model.predict_proba(Test)
isFraud = []

length= Test.shape[0]

for i in range(length):

    if probablities[i][0]>=0.5: 

        isFraud.append(0)

    elif probablities[i][0]<0.5:

        isFraud.append(1)
sns.countplot(isFraud)
thisisit = model.predict_proba(Test)[:,1]
TransactionID = Test.TransactionID

Result = pd.DataFrame({'TransactionID': TransactionID ,'isFraud':thisisit})

Result.to_csv('msubmission.csv', index = False)