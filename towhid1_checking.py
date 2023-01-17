# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

# Any results you write to the current directory are saved as output.
colsToRemove = []
for col in train.columns:
    if train[col].std() == 0:
        colsToRemove.append(col)

train.drop(colsToRemove, axis=1, inplace=True)
test.drop(colsToRemove, axis=1, inplace=True)

# remove duplicate columns
colsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    for j in range(i+1,len(columns)):
        if np.array_equal(v,train[columns[j]].values):
            colsToRemove.append(columns[j])
            
train.drop(colsToRemove, axis=1, inplace=True)
test.drop(colsToRemove, axis=1, inplace=True)
train.shape
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFpr
target=train.TARGET.values
train.drop(['ID','TARGET'],axis=1,inplace = True)
ids=test.ID.values
test.drop(['ID'],axis=1,inplace = True)
print (train.shape,'\n',test.shape)


slct = SelectFpr(alpha = 0.000001) #Filter: Select the pvalues below alpha based on a FPR test.
trainFeatures = slct.fit_transform(train, target)

print (trainFeatures.shape)

colsToRetain = slct.get_support(indices = True)

columns = train.columns
colsToRemove = []
for i in range(len(columns)):
    if i not in colsToRetain:
        colsToRemove.append(columns[i])
        print (columns[i])
testFeatures = test.drop(colsToRemove, axis=1).values
print (testFeatures.shape)

import xgboost as xgb
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=550, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)
X_train, X_test, y_train, y_test = train_test_split(trainFeatures, target, test_size=0.3)


X_fit, X_eval, y_fit, y_eval= train_test_split(trainFeatures, target, test_size=0.3)
# fitting
#clf.fit(trainFeatures, trainLabels, eval_metric="auc", early_stopping_rounds=20, eval_set=[(X_test, y_test)])


#test_pred = clf.predict_proba(testFeatures)[:,1]

#submission = pd.DataFrame({"ID":ids, "TARGET":test_pred})
#submission.to_csv("submission-fpr.csv", index=False)


# classifier
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4,
                        subsample=0.95, colsample_bytree=0.85, seed=4242)
# fitting
clf.fit(trainFeatures, target, early_stopping_rounds=365, eval_metric="auc", eval_set=[(X_eval, y_eval)])



from sklearn.metrics import roc_auc_score
print('Overall AUC:', roc_auc_score(target, clf.predict_proba(trainFeatures)[:,1]))
y_pred= clf.predict_proba(testFeatures)[:,1]

submission = pd.DataFrame({"ID":ids, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)
print ('done !')
