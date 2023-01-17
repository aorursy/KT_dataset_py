import os

import pandas as pd

import numpy as np

import matplotlib.pylab as plt

%matplotlib inline



print(os.listdir("../input/loan-data-cleaned"))
loan = pd.read_csv('../input/loan-data-cleaned/cleanedData.csv')

r,c=loan.shape

print(f"The number of rows {r}\nThe number of columns {c}")
loan.head(5)
loan.dropna(axis=0, how = 'any', inplace = True)

r1,c1=loan.shape

print(f"The difference between earlier and dropped Nan rows: {r-r1}")
mask = (loan.loan_status == 'Charged Off')

loan['target'] = 0

loan.loc[mask,'target'] = 1
del loan['loan_status']
loan.loc[loan['target']==1]
loan.loc[loan['target']==0]
loan.dtypes
categorical = loan.columns[loan.dtypes == 'object']

categorical
X = pd.get_dummies(loan[loan.columns], columns=categorical).astype(float)

y = loan['target']
X
if 'target' in X:

    del X['target']

X.columns
y
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE



X_scaled = preprocessing.scale(X)

(X_scaled)

print('   ')

print(X_scaled.shape)
def run_models(X_train, y_train, X_test, y_test, model_type = 'Non-balanced'):

    

    clfs = {'GradientBoosting': GradientBoostingClassifier(verbose=1,max_depth= 6, n_estimators=100, max_features = 0.3),

            'LogisticRegression' : LogisticRegression(verbose=1),

            #'GaussianNB': GaussianNB(),

            'RandomForestClassifier': RandomForestClassifier(verbose=1,n_estimators=10)

            }

    cols = ['model','matthews_corrcoef', 'roc_auc_score', 'precision_score', 'recall_score','f1_score']



    models_report = pd.DataFrame(columns = cols)

    conf_matrix = dict()



    for clf, clf_name in zip(clfs.values(), clfs.keys()):



        clf.fit(X_train, y_train)



        y_pred = clf.predict(X_test)

        y_score = clf.predict_proba(X_test)[:,1]



        print('computing {} - {} '.format(clf_name, model_type))



        tmp = pd.Series({'model_type': model_type,

                         'model': clf_name,

                         'roc_auc_score' : metrics.roc_auc_score(y_test, y_score),

                         'matthews_corrcoef': metrics.matthews_corrcoef(y_test, y_pred),

                         'precision_score': metrics.precision_score(y_test, y_pred),

                         'recall_score': metrics.recall_score(y_test, y_pred),

                         'f1_score': metrics.f1_score(y_test, y_pred)})



        models_report = models_report.append(tmp, ignore_index = True)

        conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)



        plt.figure(1, figsize=(6,6))

        plt.xlabel('false positive rate')

        plt.ylabel('true positive rate')

        plt.title('ROC curve - {}'.format(model_type))

        plt.plot(fpr, tpr, label = clf_name )

        plt.legend(loc=2, prop={'size':11})

    plt.plot([0,1],[0,1], color = 'black')

    

    return models_report, conf_matrix
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.4, random_state=0)

models_report, conf_matrix = run_models(X_train, y_train, X_test, y_test, model_type = 'Non-balanced')
models_report
conf_matrix['LogisticRegression']
index_split = int(len(X)/2)

X_train, y_train = SMOTE().fit_sample(X_scaled[0:index_split, :], y[0:index_split])

X_test, y_test = X_scaled[index_split:], y[index_split:]



#scores = cross_val_score(clf, X_scaled, y , cv=5, scoring='roc_auc')



models_report_bal, conf_matrix_bal = run_models(X_train, y_train, X_test, y_test, model_type = 'Balanced')
models_report_bal
conf_matrix_bal['LogisticRegression']