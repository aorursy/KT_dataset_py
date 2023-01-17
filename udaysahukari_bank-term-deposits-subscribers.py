import numpy as np

import pandas as pd



from pandas import DataFrame,Series



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import r2_score
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/bank-term-deposit-subscribers/bank-additional-full.csv', sep = ';')
df.head()
df.isnull().sum()
df.shape
df.info()
df.describe()
print('Number of unique values in each column:')

for col in df.columns[0:]:

    print(col,':')

    print('nunique =', df[col].nunique())

    print('unique =', df[col].unique())

    print()
df['y'] = df['y'].replace({'yes':1,'no':0})
# Label Encoding



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
cols = df.select_dtypes(object).columns



for i in cols:

    df[i] = le.fit_transform(df[i])
y = df['y']

x = df.drop(['y'], axis = 1)
y.head()
x.head()
# split into train and test



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.7, random_state=1)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
# Visualize the Y variable for oversampling check - bi class



sns.countplot(df['y'])

plt.show()
term_dep_subs = len(df[df['y'] == 1])

no_term_dep_subs = len(df[df['y'] == 0])

total = term_dep_subs + no_term_dep_subs



term_dep_subs = (term_dep_subs / total) * 100

no_term_dep_subs = (no_term_dep_subs / total) * 100



print('term_dep_subs:',term_dep_subs)

print('no_term_dep_subs:',no_term_dep_subs)
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix, roc_curve,roc_auc_score,r2_score
def disp_confusion_matrix(model, x, y):

    ypred = model.predict(x)

    cm = confusion_matrix(y,ypred)

    ax = sns.heatmap(cm,annot=True,fmt='d')



    ax.set_xlabel('Predicted labels')

    ax.set_ylabel('True Labels')

    ax.set_title('Confusion Matrix')

    plt.show()

    

    tp = cm[1,1]

    fn = cm[1,0]

    fp = cm[0,1]

    tn = cm[0,0]

    accuracy = (tp+tn)/(tp+fn+fp+tn)

    precision = tp/(tp+fp)

    recall = tp/(tp+fn)

    f1 = (2*precision*recall)/(precision+recall)

    print('Accuracy =',accuracy)

    print('Precision =',precision)

    print('Recall =',recall)

    print('F1 Score =',f1)
def disp_roc_curve(model, xtest, ytest):

    yprob = model.predict_proba(xtest)

    fpr,tpr,threshold = roc_curve(ytest,yprob[:,1])

    roc_auc = roc_auc_score(ytest,yprob[:,1])



    print('ROC AUC =', roc_auc)

    plt.figure()

    lw = 2

    plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC Curve (area = %0.2f)'%roc_auc)

    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('FPR')

    plt.ylabel('TPR')

    plt.title('ROC Curve')

    plt.legend(loc='lower right')

    plt.show()
#from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

#from imblearn.over_sampling import SMOTE

# from imblearn.under_sampling import NearMiss

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

# from collections import Counter

# from sklearn.model_selection import KFold, StratifiedKFold

import scipy.stats as st
y = df['y']

x = df.drop(['y'], axis = 1)
import statsmodels.api as sm
X_sm = x

X_sm = sm.add_constant(X_sm)

lm = sm.Logit(y,X_sm).fit()

lm.summary()
# Base Model



from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train,y_train)



y_pred = logreg.predict(x_test)



print('Training set score = {:.3f}'.format(logreg.score(x_train,y_train)))



print('Test set score = {:.3f}'.format(logreg.score(x_test,y_test)))

#print("R squared: {}".format(r2_score(y_true=y_test,y_pred=y_pred)))
print(classification_report(y_test, y_pred))
disp_confusion_matrix(logreg, x_test, y_test)

disp_roc_curve(logreg, x_test, y_test)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)



print('Training score =', dt.score(x_train, y_train))

print('Test score =', dt.score(x_test, y_test))

#print("R squared: {}".format(r2_score(y_true=y_test,y_pred=y_pred)))
print(classification_report(y_test, y_pred))
disp_confusion_matrix(dt, x_test, y_test)

disp_roc_curve(dt, x_test, y_test)
# knn 5 default



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)



print('Training score =', knn.score(x_train, y_train))

print('Test score =', knn.score(x_test, y_test))

#print("R squared: {}".format(r2_score(y_true=y_test,y_pred=y_pred)))
print(classification_report(y_test, y_pred))
disp_confusion_matrix(knn, x_test, y_test)

disp_roc_curve(knn, x_test, y_test)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)

y_pred = nb.predict(x_test)



print('Training score =', nb.score(x_train, y_train))

print('Test score =', nb.score(x_test, y_test))

#print("R squared: {}".format(r2_score(y_true=y_test,y_pred=y_pred)))
print(classification_report(y_test, y_pred))
disp_confusion_matrix(knn, x_test, y_test)

disp_roc_curve(knn, x_test, y_test)