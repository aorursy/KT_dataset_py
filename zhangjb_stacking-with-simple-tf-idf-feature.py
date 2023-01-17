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
train_org = pd.read_csv("../input/train.csv")
test_org = pd.read_csv("../input/test.csv")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(train_org["question_text"].values.tolist())
#print(vectorizer.get_feature_names())
print(X.shape)
Y = train_org.target
test_X = vectorizer.transform(test_org.question_text.values.tolist())
from sklearn.model_selection import StratifiedKFold
kfold_indexes = []
kfold = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
all_val_index = []
for i, (train_index, valid_index) in enumerate(kfold.split(X, Y)):
    kfold_indexes.append([train_index, valid_index])
    all_val_index.extend(valid_index)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
def return_org_index(pred_shuf):
    pred = np.zeros(len(all_val_index))
    for i, id_ in enumerate(all_val_index):
        pred[id_] = pred_shuf[i]
    return pred
def one_model_predict(clf,kfold_indexes):
    pred_train = []
    train_true_label = []
    pred_test = []
    for train_index,valid_index in kfold_indexes:
        X_train, X_val, Y_train, Y_val = X[train_index], X[valid_index], Y[train_index], Y[valid_index]
        #clf = MultinomialNB(alpha=0.01)
        clf.fit(X_train, Y_train)
        pred_val = clf.predict_proba(X_val)
        pred_train.extend(pred_val)
        train_true_label.extend(Y_val)
        pred_test_one_fold = clf.predict_proba(test_X)
        pred_test.append(pred_test_one_fold)
    train_true_label = np.array(train_true_label)
    pred_train = np.array(pred_train)[:,1]
    pred_test = np.mean(pred_test,axis=0)[:,1]
    train_true_label = return_org_index(train_true_label)
    pred_train = return_org_index(pred_train)
    return train_true_label,pred_train,pred_test

clf = MultinomialNB(alpha=0.01)
train_true_label1,pred_train1,pred_test1 = one_model_predict(clf,kfold_indexes)
clf = LogisticRegression()
train_true_label2,pred_train2,pred_test2 = one_model_predict(clf,kfold_indexes)

np.shape(pred_train1)
first_layer_pred_train = np.vstack((pred_train1,pred_train2)).T
first_layer_pred_test = np.vstack((pred_test1,pred_test2)).T
np.shape(first_layer_pred_train)
from sklearn.linear_model import Ridge
# stacking
train_second_pred = []
test_second_pred =[]
train_second_true_label = []
for id_train,id_valid in kfold_indexes:
    train_onelabel_onedire = []
    Xvalid = first_layer_pred_train[id_valid]
    Xtrain = first_layer_pred_train[id_train]
    Y_train = Y[id_train]
    Y_valid = Y[id_valid]
    model = Ridge(alpha=2.0)
    model.fit(Xtrain, Y_train)
    train_second_pred.extend(model.predict(Xvalid))
    test_second_pred.append(model.predict(first_layer_pred_test))
    train_second_true_label.extend(Y_valid)

from sklearn.metrics import f1_score
pred_second_test = np.mean(test_second_pred,axis=0)
def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    def mf(x):
        p2  =  (p > x).astype(np.int)
        score = f1_score(y, p2)#fbeta_score(np.array(y), np.array(p2), beta=2, average='samples')
        return score
    x = 0.2
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
        i2 /= 1.0 * resolution
        x = i2
        score = mf(x)
        if score > best_score:
            best_i2 = i2
            best_score = score
    x = best_i2
    if verbose:
        print(best_i2, best_score)
    return best_i2
threhold = optimise_f2_thresholds(np.array(train_second_true_label), np.array(train_second_pred))
print(threhold)
sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = (pred_second_test > threhold).astype(np.int)
sub.to_csv("submission.csv", index=False)
