# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train_target = train['label']

train_features = train.iloc[:,1:].values
# used for internal testing



from sklearn import model_selection

from sklearn import ensemble

from sklearn import metrics



internal_train, internal_test = model_selection.train_test_split(train, test_size = 0.2)



internal_train_target = internal_train['label']

internal_train_features = internal_train.iloc[:,1:].values



internal_test_target = internal_test['label']

internal_test_features = internal_test.iloc[:,1:].values





rfc = ensemble.RandomForestClassifier(n_estimators=300, min_samples_split=3, n_jobs=-1)

rfc_internal_pred = rfc.fit(internal_train_features, internal_train_target).predict(internal_test_features)



conf_matrix = metrics.confusion_matrix(internal_test_target, rfc_internal_pred, labels = [0,1,2,3,4,5,6,7,8,9])

print(conf_matrix)



correct = 0

for i in list(range(10)):

    correct += conf_matrix[i][i]

r = correct / len(internal_test_target)



print('Accuracy rate: %f' % r)

    



cv_score = model_selection.cross_val_score(rfc, train_features, train_target, cv=5)



print('Cross Validation scores:')

print(cv_score)

print('Cross Validation score mean:')

print(cv_score.mean())
rfc_pred = rfc.fit(train_features, train_target).predict(test)



results = pd.DataFrame(data=rfc_pred, columns=['Label'])

results['ImageId'] = range(1, len(results) + 1)



results = results.set_index('ImageId')



results.to_csv('randomforest.csv')