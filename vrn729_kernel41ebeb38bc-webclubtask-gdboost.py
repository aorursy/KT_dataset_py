import os

print((os.listdir('../input/')))

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, f1_score

from sklearn.model_selection import KFold, cross_val_score, cross_validate, StratifiedKFold, GridSearchCV, learning_curve



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

#from sklearn.experimental import enable_hist_gradient_boosting

#from sklearn.ensemble import HistGradientBoostingClassifier

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



import math

from math import log as log
train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
test_index=test['Unnamed: 0'] #copying test index for later
train.head()
train.describe()
train.info()
test.info()
# CORRELATION

corr_matrix=train.corr()

corr_matrix['Class'].abs().sort_values()
def bar_chart(feature, var=True):

    one = train[train['Class']==1][feature].value_counts()

    zero = train[train['Class']==0][feature].value_counts()

    df = pd.DataFrame([one,zero])

    df.index = ['one','zero']

    df.plot(kind='bar', stacked=var, figsize=(10,5))
def plot_facet_train(feature, a ,b, hue = 'Class'):

    facet = sns.FacetGrid(train, hue = hue, aspect = 5)

    facet.map(sns.kdeplot, feature, shade = True)

    facet.set(xlim = (train[feature].min(), train[feature].max()))

    facet.add_legend()

    plt.xlim(a,b)

    

def plot_facet_test(feature, a ,b):

    facet = sns.FacetGrid(test, aspect = 5)

    facet.map(sns.kdeplot, feature, shade = True)

    facet.set(xlim = (test[feature].min(), test[feature].max()))

    facet.add_legend()

    plt.xlim(a,b)
plot_facet_train('V12',train.V12.min(),train.V12.max())

plot_facet_train('V12',train.V12.min(),410)

plot_facet_train('V12',410,1400)
plot_facet_test('V12',test.V12.min(),test.V12.max())
bar_chart('V2',False)

bar_chart('V3',False)

bar_chart('V4',False)

bar_chart('V5',False)

bar_chart('V7',False)

bar_chart('V8',False)

bar_chart('V9',False)

bar_chart('V11',False)

bar_chart('V16',False)
plot_facet_train('V14',train.V14.min(),train.V14.max())
plot_facet_train('V1',train.V1.min(),train.V1.max())

plot_facet_train('V6',train.V6.min(),train.V6.max())

plot_facet_train('V10',train.V10.min(),train.V10.max())

plot_facet_train('V12',train.V12.min(),train.V12.max())

plot_facet_train('V13',train.V13.min(),train.V13.max())

plot_facet_train('V14',train.V14.min(),train.V14.max())

plot_facet_train('V15',train.V15.min(),train.V15.max())
train['V19'] = 0

for i in range (0,len(train['V12'])):

    if train['V12'][i]>=1400:

        train['V19'][i] = 2

    elif train['V12'][i]>=410:

        train['V19'][i] = 1

        

test['V19'] = 0

for i in range (0,len(test['V12'])):

    if test['V12'][i]>=1400:

        test['V19'][i] = 2

    elif test['V12'][i]>=410:

        test['V19'][i] = 1
key = 'V19'

print(train[key].value_counts())

print(train[key].value_counts().count())

print('-'*40)

print(test[key].value_counts())

print(test[key].value_counts().count())
train['V20'] = 1

for i in range (0,len(train['V14'])):

    if train['V14'][i]==-1:

        train['V20'][i] = 0

        

test['V20'] = 1

for i in range (0,len(test['V14'])):

    if test['V14'][i]==-1:

        test['V20'][i] = 0
key = 'V20'

print(train[key].value_counts())

print(train[key].value_counts().count())

print('-'*40)

print(test[key].value_counts())

print(test[key].value_counts().count())
new2 = train.drop(['Unnamed: 0','Class','V14','V15','V5'],axis=1)

y = train['Class']

use_test = test.drop(['V14','V15','V5'],axis=1)

#new2.V15 = np.cbrt(np.log1p(new2.V15))

#new['V12'] = np.log1p(new['V12'])

new2.V6 = np.cbrt(new2.V6)

new2.V1 = np.log1p(new2.V1)

use_test.V6 = np.cbrt(use_test.V6)

use_test.V1 = np.log1p(use_test.V1)

#use_test.V15 = np.cbrt(np.log1p(use_test.V15))

#use_test['V12'] = np.log1p(use_test['V12'])
new2
# data split in train, test sets

new2_train,new2_test,new2y_train,new2y_test = train_test_split(new2,y,test_size=0.2,random_state=25,shuffle=True)
# CORRELATION

new2['Class'] = y

corr_matrix=new2.corr()

corr_matrix['Class'].abs().sort_values()
new2 = new2.drop(['Class'],axis=1)
gb = GradientBoostingClassifier(n_estimators=2300,max_depth=2,random_state=25,max_features='sqrt',

                               learning_rate=0.05,min_samples_split=2,min_samples_leaf=3,

                               subsample=0.95)
gb.fit(new2_train,new2y_train)

gb_pred = gb.predict(new2_test)

print(classification_report(new2y_test,gb_pred))

print(confusion_matrix(new2y_test,gb_pred))

print(accuracy_score(new2y_test,gb_pred))

print(roc_auc_score(new2y_test,gb_pred))
skfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 25)

print(cross_val_score(gb,new2,y,cv=skfold,scoring='f1_macro'))
gb.fit(new2,y)
predictors = [x for x in new2.columns]

feat_imp = pd.Series(gb.feature_importances_, predictors).sort_values(ascending=False)

feat_imp.plot(kind='bar', title='Feature Importances')

plt.ylabel('Feature Importance Score')
df_test = use_test.loc[:,'V1':'V20']

pred = gb.predict_proba(df_test)
result=pd.DataFrame()

result['Id'] = test['Unnamed: 0']

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('new19.csv',index=False)