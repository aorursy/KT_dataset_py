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
# Train_test split & Feature selection



from sklearn import model_selection

from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA

from sklearn import metrics

import time



internal_train, internal_test = model_selection.train_test_split(train, test_size = 0.2)



internal_train_target = internal_train['label']

internal_train_features = internal_train.iloc[:,1:].values



internal_test_target = internal_test['label']

internal_test_features = internal_test.iloc[:,1:].values



start_time = time.time()



# PCA - dimensionality reduction

pca = PCA(n_components=400)

internal_train_features = pca.fit_transform(internal_train_features)

internal_test_features = pca.transform(internal_test_features)



print('Running time: %f seconds' % (time.time()-start_time))
# Internal testing



start_time = time.time()



# MLP Classifier

clf = MLPClassifier(hidden_layer_sizes=(300,), solver='sgd', learning_rate='adaptive')

clf_internal_pred = clf.fit(internal_train_features, internal_train_target).predict(internal_test_features)



conf_matrix = metrics.confusion_matrix(internal_test_target, clf_internal_pred, labels = [0,1,2,3,4,5,6,7,8,9])

print(conf_matrix)



correct = 0

for i in list(range(10)):

    correct += conf_matrix[i][i]

accuracy = correct / len(internal_test_target)

    

print('Accuracy rate: %f' % accuracy)



print('Running time: %f seconds' % (time.time()-start_time))
# Cross validation



from sklearn.pipeline import make_pipeline



start_time = time.time()



train_target = train['label']

train_features = train.iloc[:,1:].values



clf = make_pipeline(PCA(n_components=400), MLPClassifier(hidden_layer_sizes=(300,), solver='sgd', learning_rate='adaptive'))



print('Running time: %f seconds' % (time.time()-start_time))

    

cv_score = model_selection.cross_val_score(clf, train_features, train_target, cv=3)



print('Cross Validation scores:')

print(cv_score)

print('Cross Validation score mean:')

print(cv_score.mean())



print('Running time: %f seconds' % (time.time()-start_time))
clf_pred = clf.fit(train_features, train_target).predict(test)



results = pd.DataFrame(data=clf_pred, columns=['Label'])

results['ImageId'] = range(1, len(results) + 1)



results = results.set_index('ImageId')
results.to_csv('mlpclassifier.csv')