import numpy as np 

import pandas as pd 

import os

import itertools

import seaborn as sns

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix
cc = pd.read_csv("../input/creditcardfraud/creditcard.csv")
sns.countplot(cc.Class);

# Data is highly imbalanced
Correalation_Matrix = cc[cc.columns[1:]].corr()['Class'][:]

Correlation_Matrix = pd.DataFrame(Correalation_Matrix)

Correlation_Matrix.Class.plot(figsize=(10,5))

plt.ylabel('Correlation Score')

plt.xlabel('Features')
cc.corrwith(cc.Class).plot.bar(

        figsize = (20, 10), title = "Correlation with class", fontsize = 15,

        rot = 45, grid = True)
#price range correlation

corr=cc.corr()

corr=corr.sort_values(by=["Class"],ascending=False).iloc[0].sort_values(ascending=False)

plt.figure(figsize=(15,20))

sns.barplot(x=corr.values, y =corr.index.values);

plt.title("Correlation Plot")
Column = abs(Correalation_Matrix)

Column = pd.DataFrame(Column)

Column = Column.reset_index()

Column.rename(columns={'index':'Features','Class':'Correlation_score'}, inplace=True)

Column.head(5)
relevant = []

for i in range(len(Column)):

    if Column.Correlation_score[i] > 0.05:

           relevant.append(Column.Features[i])

    else:

        continue
def keep_cols(DataFrame, keep_these):

    drop_these = list(set(list(DataFrame)) - set(keep_these))

    return DataFrame.drop(drop_these, axis = 1)



DF = cc.pipe(keep_cols, relevant)

DF.head()
plt.figure(figsize=(10,8))

cor = DF.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
y = cc.Class

cc.drop("Class", inplace=True, axis=1)

cc.head()
X_train, X_test, Y_train, Y_test = train_test_split(cc, y, test_size = 0.3, random_state=2019, stratify = y)
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                              colsample_bynode=1, colsample_bytree=0.9, gamma=0,

                              learning_rate=0.1, max_delta_step=0, max_depth=10,

                              min_child_weight=1, missing=None, n_estimators=500, n_jobs=-1,

                              nthread=None, objective='binary:logistic', random_state=0,

                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

                              silent=None, subsample=0.9, verbosity=0)
model.fit(X_train, Y_train)

y_pred=model.predict(X_test)
metrics.accuracy_score(Y_test, y_pred)*100
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

confusion_mtx = confusion_matrix(Y_test, y_pred) 

plot_confusion_matrix(confusion_mtx, classes = range(2)) 