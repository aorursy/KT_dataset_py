# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")
df_test=pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")
df_test.iloc[:,1:201]=df_test.iloc[:,1:201].astype(np.float32)
df_test.info()
df_train.iloc[:,2:202]=df_train.iloc[:,2:202].astype(np.float32)
df_train.info()
df_train.head()
df_test.head()
df_test.isna().values.any()
df_train.isna().values.any()
df_train.describe()
df_test.describe()
li=['var_0', 'var_1','var_2','var_3', 'var_4', 'var_5', 'var_6', 'var_7', 'var_8', 'var_9', 'var_10','var_11','var_12', 'var_13', 'var_14', 'var_15']
fig, ax = plt.subplots(4,4,figsize=(14,14))

i=0

for il in li:

    i+=1

    plt.subplot(4,4,i)

    plt.scatter(df_test[il],df_train[il],marker="*",alpha=0.6)



plt.tight_layout()

plt.show()
df_train["target"].value_counts()
df_train.head()
cor=df_train.corr()



cor_target = abs(cor["target"])



for i in li:

    relevant_features = cor_target[cor_target>0.3]

    relevant_features
to_drop=["target","ID_code"]
X=df_train.drop(to_drop,1)

y=df_train["target"]
X
reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)
imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
imp_coef[imp_coef>0.02]
imp_coef[imp_coef<-0.02]
to_drope=imp_coef[(imp_coef>-0.000625) & (imp_coef<0.000625)].index.tolist()
X_trains,X_test,y_trains,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
logreg = LogisticRegression(C=1, random_state=42)

logreg.fit(X_trains, y_trains)
y_pred = logreg.predict(X_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test, y_pred)

cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)



cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=[5,2])



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
to_drop_beta=["target","ID_code"]

to_drop_beta=to_drope+to_drop_beta

X_beta=df_train.drop(to_drop_beta,1)

y_beta=df_train["target"]
X_trains_b,X_test_b,y_trains_b,y_test_b=train_test_split(X_beta,y_beta,test_size=0.2,random_state=42,stratify=y_beta)
logreg = LogisticRegression(C=1, random_state=42)

logreg.fit(X_trains_b, y_trains_b)
y_pred_b = logreg.predict(X_test_b)
accuracy_score(y_test_b,y_pred_b)
print(classification_report(y_test_b,y_pred_b))
logit_roc_auc = roc_auc_score(y_test_b, logreg.predict(X_test_b))

fpr, tpr, thresholds = roc_curve(y_test_b, logreg.predict_proba(X_test_b)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
clf = svm.SVC(kernel='linear',decision_function_shape='ovr')
clf = GaussianNB()

clf.fit(X_trains_b, y_trains_b)
y_pred_b = clf.predict(X_test_b)
accuracy_score(y_test_b,y_pred_b)
print(classification_report(y_test_b,y_pred_b))
cm=confusion_matrix(y_test_b, y_pred_b)

cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)



cm = pd.DataFrame(cm, index=np.unique(y_test_b), columns=np.unique(y_test_b))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=[5,2])



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)
logit_roc_auc = roc_auc_score(y_test_b, clf.predict(X_test_b))

fpr, tpr, thresholds = roc_curve(y_test_b, clf.predict_proba(X_test_b)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
#submission_nb = pd.DataFrame({

#    "ID_code": df_test["ID_code"],

#    "target": y_pred_b

#})

#submission_nb.to_csv('naive_baise_submission.csv', index=False)
from sklearn.ensemble import RandomForestRegressor

from sklearn import tree

dt = tree.DecisionTreeClassifier(random_state = 42)

rf = RandomForestRegressor(n_estimators = 15, random_state = 42)

dt.fit(X_trains_b, y_trains_b)
y_pred_b = dt.predict(X_test_b)
accuracy_score(y_test_b,y_pred_b)
print(classification_report(y_test_b,y_pred_b))
cm=confusion_matrix(y_test_b, y_pred_b)

cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)



cm = pd.DataFrame(cm, index=np.unique(y_test_b), columns=np.unique(y_test_b))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=[5,2])



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)
logit_roc_auc = roc_auc_score(y_test_b, dt.predict(X_test_b))

fpr, tpr, thresholds = roc_curve(y_test_b, dt.predict_proba(X_test_b)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()

from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_trains_b, y_trains_b)
y_pred_b = model.predict(X_test_b)
accuracy_score(y_test_b,y_pred_b)
print(classification_report(y_test_b,y_pred_b))
cm=confusion_matrix(y_test_b, y_pred_b)

cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)



cm = pd.DataFrame(cm, index=np.unique(y_test_b), columns=np.unique(y_test_b))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=[5,2])



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)
logit_roc_auc = roc_auc_score(y_test_b, model.predict(X_test_b))

fpr, tpr, thresholds = roc_curve(y_test_b, model.predict_proba(X_test_b)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
testik=df_test.drop(['ID_code'],axis=1)

testik
testik.shape
testik=testik.drop(to_drope,1)
cde=df_test[["ID_code"]]

cde=cde.values.reshape(200000,).shape
logreg.predict(testik.values).shape
submission_log=pd.DataFrame(

{

    "ID_code":df_test[["ID_code"]].values.reshape(200000,),

    "target":logreg.predict(testik)

}

)

submission_log["target"].value_counts()

submission_log.to_csv("Logreg.csv",index=False)
submission_bayise=pd.DataFrame(

{

    "ID_code":df_test[["ID_code"]].values.reshape(200000,),

    "target":clf.predict(testik)

}

)

submission_bayise["target"].value_counts()

submission_bayise.to_csv("Bayise.csv",index=False)
submission_xg=pd.DataFrame(

{

    "ID_code":df_test[["ID_code"]].values.reshape(200000,),

    "target":model.predict(testik)

}

)

submission_xg["target"].value_counts()

submission_xg.to_csv("Xgboost.csv",index=False)
submission_tree=pd.DataFrame(

{

    "ID_code":df_test[["ID_code"]].values.reshape(200000,),

    "target":dt.predict(testik)

}

)

submission_tree["target"].value_counts()

submission_tree.to_csv("tree.csv",index=False)