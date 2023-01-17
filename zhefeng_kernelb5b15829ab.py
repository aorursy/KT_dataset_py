# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline



## data analysis trio :)

import matplotlib.pyplot as plt # plot

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns; sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/winequality-red.csv')



print(df.shape)



df.head()
_tmp = df.quality.value_counts().sort_values()

plt.barh(_tmp.index, _tmp)



plt.show()



_tmp = df.quality.value_counts(normalize=True).sort_values()

print(_tmp)
ml_df = df.copy() # In general it would be good to not touch the originial raw data frame :)

ml_df['is_good'] = df.quality.apply(lambda q: True if q>=7 else False)



_tmp = ml_df.is_good.value_counts().sort_values()

print("-"*10, "total good/bad", "-"*10)

print(_tmp)
sns.pairplot(ml_df, hue='is_good')

ml_df.columns



ftr_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol']


train_ratio = .8



train_num = int(train_ratio * len(df))

shuffled = ml_df.sample(len(df))



X, y = np.array(shuffled[ftr_names]), np.array(shuffled.is_good)



X_train, X_test = X[:train_num], X[train_num:]

y_train, y_test = y[:train_num], y[train_num:]



print(f"total number of training points: {len(X_train)} with {sum(y_train)} positive example")

print(f"total number of test points: {len(X_test)} with {sum(y_test)} positive example")
from sklearn.svm import SVC



lsvc= SVC(kernel='linear')

lsvc.fit(X_train, y_train)



s_train = lsvc.decision_function(X_train)

s_test = lsvc.decision_function(X_test)
from sklearn.metrics import average_precision_score, precision_recall_curve



def show_pr_curve(y_train, s_train, y_test, s_test):

    ap_train = average_precision_score(y_train, s_train)

    ap_test = average_precision_score(y_test, s_test)



    prc_train, rec_train, _ = precision_recall_curve(y_train, s_train)

    prc_test, rec_test, _ = precision_recall_curve(y_test, s_test)



    plt.figure(figsize=(8, 8))

    plt.plot(rec_train, prc_train)

    plt.plot(rec_test, prc_test)

    plt.ylabel("precision")

    plt.xlabel("recall")

    plt.legend([f'train (ap: {ap_train})', f'test (ap: {ap_test})'])

    plt.show()

show_pr_curve(y_train, s_train, y_test, s_test)

from sklearn.svm import SVC # note we are not using LinearSVC any more, as we may change the kernel later



pos_ratio = ml_df.is_good.sum()/ml_df.is_good.count()



print("positive ratio is:", pos_ratio)



class_weight = {True: 1/pos_ratio, False: 1}

svc = SVC(kernel='linear', class_weight=class_weight)



svc.fit(X_train, y_train)



s_train = svc.decision_function(X_train)

s_test = svc.decision_function(X_test)



show_pr_curve(y_train, s_train, y_test, s_test)



from random import sample



neg_ind = list(np.where(y_train==False)[0])

pos_ind = list(np.where(y_train==True)[0])



print(f"there are in total {len(neg_ind)} negatives")

print(f"there are in total {len(pos_ind)} positive")



neg_size = len(pos_ind)



downsampled_ind = sample(neg_ind, neg_size) + pos_ind # create a balanced indices

downsampled_ind = sample(downsampled_ind, len(downsampled_ind)) # and reshuffle it a bit



X_train_dwnsmp, y_train_dwnsmp = X_train[downsampled_ind, :], y_train[downsampled_ind]



print(f"there are in total {len(y_train_dwnsmp[y_train_dwnsmp==False])} negatives after down sample")

print(f"there are in total {len(y_train_dwnsmp[y_train_dwnsmp==True])} positive after down sample")



svc = SVC(kernel='linear')



svc.fit(X_train_dwnsmp, y_train_dwnsmp)



s_train = svc.decision_function(X_train_dwnsmp)

s_test = svc.decision_function(X_test)



show_pr_curve(y_train_dwnsmp, s_train, y_test, s_test)

from sklearn.metrics import precision_score, classification_report





svc = SVC(kernel='linear')



svc.fit(X_train, y_train)

y_pred_test = svc.predict(X_test)

print("without down sample:")

print(classification_report(y_test, y_pred_test))

print("="*20)



svc.fit(X_train_dwnsmp, y_train_dwnsmp)

y_pred_test = svc.predict(X_test)

print("with down sample:")

print(classification_report(y_test, y_pred_test))

print("="*20)





from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline



clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))



clf.fit(X_train_dwnsmp, y_train_dwnsmp)



s_train = clf.decision_function(X_train_dwnsmp)

s_test = clf.decision_function(X_test)



show_pr_curve(y_train_dwnsmp, s_train, y_test, s_test)

clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto', C=2))



clf.fit(X_train, y_train)



s_train = clf.decision_function(X_train)

s_test = clf.decision_function(X_test)



show_pr_curve(y_train, s_train, y_test, s_test)

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate



params = {

    'gamma': np.arange(0.1, 1, 0.1),

    'C': np.arange(2, 3, .1),

}



stds = StandardScaler()



clf = SVC()

grid = GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring='average_precision')



grid.fit(stds.fit_transform(X_train), y_train)



clf =make_pipeline(stds, grid.best_estimator_)

print("best estimator:", clf)





s_train = clf.decision_function(X_train)

s_test = clf.decision_function(X_test)



show_pr_curve(y_train, s_train, y_test, s_test)



print(cross_val_score(clf, X_train, y_train, cv=5, scoring='average_precision'))

from sklearn.ensemble import RandomForestClassifier



clf =RandomForestClassifier()



clf.fit(X_train, y_train)





s_train = clf.predict_proba(X_train)[:, 1]

s_test = clf.predict_proba(X_test)[:, 1]



show_pr_curve(y_train, s_train, y_test, s_test)



print(cross_val_score(clf, X_train, y_train, cv=5, scoring='average_precision'))



params = {

    "n_estimators": np.arange(10, 100, 5),

#     "min_samples_split": np.arange(2, 20, 5),

#     "min_samples_leaf": np.arange(1, 20, 5),

}



clf =RandomForestClassifier(min_samples_split=10, n_estimators=50)

grid = GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring='average_precision')



grid.fit(X_train, y_train)



clf = grid.best_estimator_

print("best estimator:", clf)



s_train = clf.predict_proba(X_train)[:, 1]

s_test = clf.predict_proba(X_test)[:, 1]



show_pr_curve(y_train, s_train, y_test, s_test)



print(cross_val_score(clf, X_train, y_train, cv=5, scoring='average_precision'))



print(cross_val_score(clf, X_train, y_train, cv=5))
from tpot import TPOTClassifier



tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2,

                      config_dict='TPOT light', scoring='average_precision')



tpot.fit(X_train, y_train)



print(tpot.score(X_test, y_test))

tpot.export('tpot_wine_pipeline.py')



! cat tpot_wine_pipeline.py

import numpy as np

import pandas as pd

from sklearn.feature_selection import SelectPercentile, f_classif

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler



# Average CV score on the training set was:0.7156702107386644

exported_pipeline = make_pipeline(

    VarianceThreshold(threshold=0.0001),

    MinMaxScaler(),

    KNeighborsClassifier(n_neighbors=26, p=2, weights="distance")

)



exported_pipeline.fit(X_train, y_train)







y_pred_train = exported_pipeline.predict(X_train)

y_pred_test = exported_pipeline.predict(X_test)



s_train = exported_pipeline.predict_proba(X_train)[:, 1]

s_test = exported_pipeline.predict_proba(X_test)[:, 1]





print(classification_report(y_pred_train, y_train))

print(classification_report(y_pred_test, y_test))



show_pr_curve(y_train, s_train, y_test, s_test)




