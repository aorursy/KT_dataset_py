import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

data = pd.read_csv("../input/heart-disease-uci/heart.csv")
data.head()
data.target.mean()
data.columns
numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']

categorical = [ 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
data[numerical].corr().target
sns.pairplot(data[numerical].sample(frac=0.3), hue='target');

stats.ttest_ind(data[data.chol > data.chol.mean()].target,data[data.chol < data.chol.mean()].target)
stats.ttest_ind(data[data.chol > data.chol.median()].target,data[data.chol < data.chol.mean()].target)
for column in categorical:

    print(data.groupby(column).target.agg(['mean', 'count']))
data['binned_age'] = pd.cut(data.age,20)
data.groupby('binned_age').target.mean().plot()
data.groupby('binned_age').age.count().plot()
data['thal_cat']  = data.thal==2

data['thal_cat'] =  data.thal_cat.astype('int')



data['slope_cat']  = data.slope==2

data['slope_cat'] =  data.slope_cat.astype('int')



data['cp_cat']  = data.cp==0

data['cp_cat'] =  data.cp_cat.astype('int')



#data['ca_cat']  = data.ca==0

#data['ca_cat'] =  data.ca_cat.astype('int')
relevant = data[['thalach', 'oldpeak', 'thal_cat', 'cp_cat', 'ca', 'slope_cat', 'exang']]
X_train, X_test, y_train, y_test = train_test_split(relevant, data.target, test_size=0.2)
def sensitivity_scorer(Estimator,X,y):

    y_pred = Estimator.predict(X)

    C = confusion_matrix(y,y_pred)

    TN = C[0,0]

    TP = C[1,1]

    FN = C[1,0]

    FP = C[0,1]

    return TP/(TP+FN)



def specificity_scorer(Estimator,X,y):

    y_pred = Estimator.predict(X)

    C = confusion_matrix(y,y_pred)

    TN = C[0,0]

    TP = C[1,1]

    FN = C[1,0]

    FP = C[0,1]

    return TN/(TN+FP)
classifiers = [LogisticRegression(solver='lbfgs', max_iter=300), GaussianNB(), 

               RandomForestClassifier(n_estimators=10), KNeighborsClassifier()  ]
scores = pd.DataFrame(index=classifiers, columns = ['specificity', 'spec_std', 'sensitivity', 'sens_std' ])
for clf in classifiers:



    sens = cross_val_score(clf, X_train, y_train, cv=5, scoring=sensitivity_scorer)

    spec = cross_val_score(clf, X_train, y_train, cv=5, scoring=specificity_scorer)

    scores.loc[clf,'sensitivity'] = round(sens.mean(),2)

    scores.loc[clf,'sens_std'] =  round(sens.std(),2)

    scores.loc[clf,'specificity'] = round(spec.mean(),2)

    scores.loc[clf,'spec_std'] = round(spec.std(),2)
scores
clf = LogisticRegression(solver='lbfgs', max_iter=300).fit(X_train, y_train)
y_pred = clf.predict(X_test)
C = confusion_matrix(y_test, y_pred)

TN = C[0,0]

TP = C[1,1]

FN = C[1,0]

FP = C[0,1]
C #confusion matrix
print('Specificity:', round(TN/(TN+FP),2))

print('Sensitivity:', round(TP/(TP+FN),2))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.set(xlabel='FPR', ylabel='TPR', title='ROC curve')

plt.show()
roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])