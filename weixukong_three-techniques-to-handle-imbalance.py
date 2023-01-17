# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





import matplotlib.pyplot as plt

import seaborn as sns

import scipy

import time

import random



from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import recall_score, precision_score, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, balanced_accuracy_score



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/creditcard.csv")
data.head()
data.shape
sns.countplot(data['Class'])
print("Class Ratio: ")

data.Class.value_counts() / data.shape[0] * 100
data[data.isnull().any(axis=1)]
corr = data.drop(columns = ['Class','Time']).corr()

fig, ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(corr)
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(6,5,figsize=(18,18))



    for feature in features:

        i += 1

        plt.subplot(6,5,i)

        sns.distplot(df1[feature], kde = True,label=label1)

        sns.distplot(df2[feature], kde = True,label=label2)

        plt.xlabel(feature, fontsize=11)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
d0 = data.loc[data['Class'] == 0]

d1 = data.loc[data['Class'] == 1]

features = data.drop(columns = ['Time','Class'])

plot_feature_distribution(d0, d1, 'Class: 0', 'Class: 1', features)
y = data.Class.values

X = data.drop(columns = ['Class','Time'])



features= X.columns



X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 39)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_org)

X_test = scaler.transform(X_test_org)
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    fmt = '.4f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

def get_results(y_test,y_pred):



    cnf_matrix = confusion_matrix(y_test,y_pred)

    np.set_printoptions(precision=2)



    print("Recall metric: {:<8.4f}".format(recall_score(y_test, y_pred)))

    print("Precision metric: {:<8.4f}".format(precision_score(y_test, y_pred)))

    print("Balanced_accuracy metric: {:<8.4f}".format(balanced_accuracy_score(y_test, y_pred)))



    # Plot normalized confusion matrix

    class_names = [0,1]

    plt.figure()

    plot_confusion_matrix(cnf_matrix,

                          normalize=True,

                          classes=class_names,

                          title='Confusion matrix')

    plt.show()
def cross_val(model, X_train, y_train, DECISION_FUNCTIONS = {"RidgeClassifier", "SGDClassifier"}):

    folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=2019)

    oof_proba = np.zeros(len(X_train))

    oof_class = np.zeros(len(X_train))



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):

        X_trn = X_train[trn_idx]

        X_val = X_train[val_idx]

        y_trn = y_train[trn_idx]

        y_val = y_train[val_idx]

        

        model.fit(X_trn,y_trn)

        name = type(model).__name__

        

        if name in DECISION_FUNCTIONS:

            proba = model.decision_function(X_val)

        else:

            proba = model.predict_proba(X_val)[:, 1]

        

        oof_proba[val_idx] = proba

        oof_class[val_idx] = model.predict(X_val)

    

    return oof_proba,oof_class
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier



models = []



models.append(LogisticRegression(C =0.01, solver = 'lbfgs', max_iter = 1000, class_weight = 'balanced'))

models.append(RidgeClassifier(class_weight = 'balanced'))

models.append(SGDClassifier(tol=1e-3, max_iter=10000, class_weight = 'balanced'))

models.append(DecisionTreeClassifier(max_depth = 5, class_weight = 'balanced'))

models.append(RandomForestClassifier(n_estimators=100, class_weight = 'balanced'))

models.append(LGBMClassifier(n_estimators = 200, max_depth = 3, learning_rate = 0.025, class_weight = 'balanced'))



DECISION_FUNCTIONS = {"Ridge", "SGD"}
fig = plt.figure(figsize=(20,16))

ax1 = fig.add_subplot(2,2,1)

ax1.set_xlim([-0.05,1.05])

ax1.set_ylim([-0.05,1.05])

ax1.set_xlabel('Recall')

ax1.set_ylabel('Precision')

ax1.set_title('PR Curve')



ax2 = fig.add_subplot(2,2,2)

ax2.set_xlim([-0.05,1.05])

ax2.set_ylim([-0.05,1.05])

ax2.set_xlabel('False Positive Rate')

ax2.set_ylabel('True Positive Rate')

ax2.set_title('ROC Curve')

        

ax3 = fig.add_subplot(2,2,3)

ax3.set_xlim([0.8,1])

ax3.set_ylim([0,0.7])

ax3.set_xlabel('Recall')

ax3.set_ylabel('Precision')

ax3.set_title('PR Curve')

        

ax4 = fig.add_subplot(2,2,4)

ax4.set_xlim([0,0.15])

ax4.set_ylim([0.85,1.00])

ax4.set_xlabel('False Positive Rate')

ax4.set_ylabel('True Positive Rate')

ax4.set_title('ROC Curve')

oofs = []



for model,k in zip(models,'bgrcmykw'[0:len(models)]):

        

    name = type(model).__name__



    oof_proba,oof_class = cross_val(model,X_train,y_train)

    

    oofs.append(oof_class)

    

    p,r,_ = precision_recall_curve(y_train,oof_proba)

    fpr,tpr,_ = roc_curve(y_train,oof_proba)

    score = roc_auc_score(y_train, oof_proba)

    

    ax1.plot(r,p,c=k, label=name)

    ax2.plot(fpr,tpr,c=k,label=name + f": auc = {format(score,'.4f')}")

    ax3.plot(r,p,c=k, label=name)

    ax4.plot(fpr,tpr,c=k,label=name + f": auc = {format(score,'.4f')}")

        

ax1.legend(loc='lower left')

ax2.legend(loc='lower right')

ax3.legend(loc='upper right')

ax4.legend(loc='lower right')







plt.show()
for model, oof in zip(models, oofs):

    

    print(type(model).__name__)

    

    get_results(y_train,oof)
lrc = LogisticRegression(C =0.01, solver = 'lbfgs', max_iter = 1000, class_weight = 'balanced')



lrc.fit(X_train,y_train)



y_pred_lrc_cw = lrc.predict(X_test)



get_results(y_test,y_pred_lrc_cw)
lgb = LGBMClassifier(n_estimators = 200, max_depth = 3, learning_rate = 0.025, class_weight = 'balanced')



lgb.fit(X_train,y_train)



y_pred_lgb_cw = lgb.predict(X_test)



get_results(y_test,y_pred_lgb_cw)
from imblearn.ensemble import BalancedBaggingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from imblearn.ensemble import BalancedRandomForestClassifier





bbc_lrc = BalancedBaggingClassifier(LogisticRegression(C = 100, solver = 'lbfgs', max_iter = 1000), 

                                    n_estimators = 100, 

                                    replacement = True, 

                                    n_jobs=-1)



bbc_lgbm = BalancedBaggingClassifier(LGBMClassifier(n_estimators = 200, max_depth = 3, learning_rate = 0.025), 

                                    n_estimators = 100, 

                                    replacement = True, 

                                    n_jobs=-1)



bbc_dtc = BalancedBaggingClassifier(DecisionTreeClassifier(max_depth = 5), 

                                    n_estimators = 100, 

                                    replacement = True, 

                                    n_jobs=-1)





bbc_gnb = BalancedBaggingClassifier(GaussianNB(), 

                                n_estimators = 100, 

                                replacement = True, 

                                n_jobs=-1)



bbc_mlp = BalancedBaggingClassifier(MLPClassifier(max_iter=200,hidden_layer_sizes=150,batch_size = 200), 

                                    n_estimators = 100, 

                                    replacement = True, 

                                    n_jobs=-1)



bbc_models = []

bbc_models.append(bbc_lrc)

bbc_models.append(bbc_lgbm)

bbc_models.append(bbc_dtc)

bbc_models.append(bbc_gnb)
for model in bbc_models:

    print(type(model).__name__ + "_" + type(model.base_estimator).__name__)

    

    oof_proba,oof_class = cross_val(model,X_train,y_train)

    

    get_results(y_train,oof_class)
bbc_lrc.fit(X_train,y_train)



bbc_lrc_y_pred = bbc_lrc.predict(X_test)



get_results(y_test, bbc_lrc_y_pred)
bbc_lgbm.fit(X_train,y_train)



bbc_lgb_y_pred = bbc_lgbm.predict(X_test)



get_results(y_test, bbc_lgb_y_pred)
bbc_mlp.fit(X_train,y_train)



bbc_mlp_y_pred = bbc_mlp.predict(X_test)



get_results(y_test, bbc_mlp_y_pred)
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline



sm_lrc = Pipeline(steps = [('sm', SMOTE()), ('logistic', LogisticRegression(C = 0.01, solver = 'lbfgs', max_iter = 1000))])



oof_proba, oof_pred = cross_val(sm_lrc,X_train,y_train)



get_results(y_train,oof_pred)
sm_lrc.fit(X_train,y_train)



sm_lrc_y_pred = sm_lrc.predict(X_test)



get_results(y_test,sm_lrc_y_pred)