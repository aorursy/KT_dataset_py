# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cross_validation import ShuffleSplit

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/mushrooms.csv')
data.shape
data.info()
data.describe()
data.head(5)
data.keys()
for feature,col_data in data.iteritems():

    if col_data.dtype == object:

        print("{} has {}".format(feature,col_data.unique()))
y_all = data['class']

X_all = data.drop('class', axis = 1)
n_samples = len(data.index)



n_features = len(data.columns)-1



n_p = dict(data['class'].value_counts())['p']



n_e = dict(data['class'].value_counts())['e']



p_rate = float(n_p)/float(n_samples)*100



print("Total number of samples: {}".format(n_samples))

print("Number of features: {}".format(n_features))

print("Number of p: {}".format(n_p))

print("Number of e: {}".format(n_e))

print("p rate : {:.2f}%".format(p_rate))
X_all = pd.get_dummies(X_all)

X_all.head()

print(X_all.keys())

print(len(X_all.keys()))
X_train,X_test,y_train,y_test = train_test_split(X_all,y_all,test_size = 0.2,random_state = 2)
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

    return f1_score(target.values, y_pred, pos_label= 'p')





def train_predict(clf, X_train, y_train, X_test, y_test):



    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    

    train_classifier(clf, X_train, y_train)



    print("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))

    print("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier



clf_A = RandomForestClassifier(random_state = 1)

clf_B = SVC(random_state = 1)

clf_C = SGDClassifier(random_state = 1)

clf_D = CalibratedClassifierCV()

clf_E = LogisticRegression(random_state = 1)

clf_F = SGDClassifier(random_state = 1)



clf_list = [clf_A,clf_B,clf_C,clf_D,clf_E,clf_F]

for i in clf_list:

    train_predict(i, X_train, y_train, X_test, y_test)
indices = np.argsort(clf_A.feature_importances_)[::-1]



features_ranking = dict()



nb_rank = 10



for f in range(X_all.shape[1]):

    if X_all.columns[indices[f]].split('_', 1 )[0] in features_ranking.keys():

        features_ranking[X_all.columns[indices[f]].split('_', 1 )[0]] += clf_A.feature_importances_[indices[f]]

    else:

        features_ranking[X_all.columns[indices[f]].split('_', 1 )[0]] = clf_A.feature_importances_[indices[f]]

features_ranking= sorted(features_ranking.items(), key=lambda features_ranking:features_ranking[1], reverse = True)





# Print the feature ranking

print('Feature ranking:')

for i in range(nb_rank):

    print('%d. feature %s (%f)' % (i+1,features_ranking[i][0], features_ranking[i][1]))