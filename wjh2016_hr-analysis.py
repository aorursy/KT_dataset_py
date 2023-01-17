# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/HR_comma_sep.csv")

y_all = data['left']

X_all = data.drop('left', axis = 1)
data.shape
data.info()
data.describe()
data.keys()
for feature,col_data in X_all.iteritems():

    if col_data.dtype == object:

        print("{} has {}".format(feature,col_data.unique()))
n_employee = len(data.index)



n_features = len(data.columns)-1



n_left = dict(data['left'].value_counts())[1.0]



n_stay = dict(data['left'].value_counts())[0.0]



left_rate = float(n_left)/float(n_employee)*100



# 输出结果

print("Total number of employee: {}".format(n_employee))

print("Number of features: {}".format(n_features))

print("Number of employee who passed: {}".format(n_left))

print("Number of employee who failed: {}".format(n_stay))

print("left rate : {:.2f}%".format(left_rate))
def preprocess_features(X):

    output = pd.DataFrame(index = X.index)

    for feature,col_data in X.iteritems():

        if col_data.dtype == object:

            print("{} has {}".format(feature,col_data.unique()))

            col_data = pd.get_dummies(col_data)

        output = output.join(col_data)

            

    return output



X_all = preprocess_features(X_all)

print(X_all.keys())
from sklearn.cross_validation import train_test_split



X_train ,X_test , y_train , y_test = train_test_split(X_all,y_all,test_size = 0.3,random_state = 1)

print("Training set has {} samples.".format(X_train.shape[0]))

print("Testing set has {} samples.".format(X_test.shape[0]))
from time import time

from sklearn.metrics import f1_score



def train_classifier(clf, X_train, y_train):



    start = time()

    clf.fit(X_train, y_train)

    end = time()

    

    #print("Trained model in {:.4f} seconds".format(end - start))



    

def predict_labels(clf, features, target):



    start = time()

    y_pred = clf.predict(features)

    end = time()

    

    #print("Made predictions in {:.4f} seconds.".format(end - start))

    return f1_score(target.values, y_pred, pos_label= 1)





def train_predict(clf, X_train, y_train, X_test, y_test):



    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    

    train_classifier(clf, X_train, y_train)



    print("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))

    print("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



clf_A = RandomForestClassifier(random_state = 1)

clf_B = SVC(random_state = 1)



clf_list = [clf_A,clf_B]

for i in clf_list:

    train_predict(i, X_train, y_train, X_test, y_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer,f1_score

from sklearn.grid_search import GridSearchCV



parameters = {'min_samples_split':np.arange(2,4,1),'max_depth':np.arange(10,30,2),

             'n_estimators':np.arange(80,100,2)}



clf = RandomForestClassifier(random_state = 1)



f1_scorer = make_scorer(f1_score,pos_label = 1)



grid_obj = GridSearchCV(clf,parameters,scoring = f1_scorer)



grid_obj.fit(X_train,y_train)



clf = grid_obj.best_estimator_



print("Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train)))

print("Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test)))
