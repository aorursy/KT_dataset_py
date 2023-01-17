# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
a = pd.DataFrame(df.isna().sum())
a.columns = ['x']
a  = a[a['x']>0]
df =df.drop(['v2a1','v18q1','rez_esc'],axis=1)
df['meaneduc'] = df['meaneduc'].fillna('999')
df['SQBmeaned'] = df['SQBmeaned'].fillna('999')
c = df[df['meaneduc'] =='999'].index.tolist()
df['edjefe'] = df['edjefe'].T.apply(lambda x: 1 if x=='yes'else x)
df['edjefe'] = df['edjefe'].T.apply(lambda x: 0 if x=='no'else x)
df['edjefa'] = df['edjefa'].T.apply(lambda x: 1 if x=='yes'else x)
df['edjefa'] = df['edjefa'].T.apply(lambda x: 0 if x=='no'else x)
df['dependency'] = df['dependency'].T.apply(lambda x: 0 if x=='no'else x)

df['dependency'] = df['dependency'].T.apply(lambda x: 1 if x=='yes'else x)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
target= pd.DataFrame(df['Target'] )
df =df.drop(['Target'],axis = 1)
df = df.drop(['idhogar','Id'],axis = 1)
gbdt = GradientBoostingClassifier(
    init=None,
    learning_rate=0.1,
    loss='deviance',
    max_depth=3,
    max_features=None,
    max_leaf_nodes=None,
    min_samples_leaf=1,
    min_samples_split=2,
    min_weight_fraction_leaf=0.0,
    n_estimators=100,
    random_state=None,
    subsample=1.0,
    verbose=0,
    warm_start=False)
gbdt.fit(df, target['Target'])
 
 
score = gbdt.feature_importances_
df = df.loc[:, gbdt.feature_importances_>0]
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df, target['Target'], test_size=0.4, random_state=0)
test_data = pd.read_csv('../input/test.csv')
columns = df.columns
test1 = test_data[columns]
test1.shape
test1['edjefe'] = test1['edjefe'].T.apply(lambda x: 1 if x=='yes' else x)
test1['edjefe'] = test1['edjefe'].T.apply(lambda x: 0 if x=='no'else x)
test1['edjefa'] = test1['edjefa'].T.apply(lambda x: 1 if x=='yes'else x)
test1['edjefa'] = test1['edjefa'].T.apply(lambda x: 0 if x=='no'else x)
test1['dependency'] = test1['dependency'].T.apply(lambda x: 0 if x=='no'else x)
test1['dependency'] = test1['dependency'].T.apply(lambda x: 1 if x=='yes'else x)
test1.shape
test1['meaneduc'] = test1['meaneduc'].fillna('999')
test1['SQBmeaned'] = test1['SQBmeaned'].fillna('999')
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train) 
fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1], pos_label=1)
#plt.plot(fpr, tpr, color="green")
print("LogisticRegression auc = ", metrics.auc(fpr, tpr))
predict = pd.DataFrame(model.predict(test1))
predict.columns = ['Target']
predict['ID'] = test_data['Id']
predict.to_csv('submission.csv',index = False)
