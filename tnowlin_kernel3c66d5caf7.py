import numpy as np

import pandas as pd

import sklearn

from sklearn.metrics import accuracy_score

import math

from scipy.stats import uniform

from random import randint

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score,RandomizedSearchCV
X_train = pd.read_csv("../input/X_train_new.csv", header=0)



X_train.head()
y_train = pd.read_csv("../input/y_train_new.csv", header=0)

y_train
X_test = pd.read_csv("../input/X_test_new.csv", header=0)

X_test.head()
y_test = pd.read_csv("../input/y_test_new.csv", header=0)

y_test.head()


features=X_test

features=features.drop(features.columns[0], axis=1)

features=features.drop(features.columns[0],axis=1)

features=features.drop(features.columns[1],axis=1)







maximum=features.groupby(['series_id']).max()

maximum.columns = [str(col) + '_max' for col in maximum.columns]





minimum=features.groupby(['series_id']).min()

minimum.columns = [str(col) + '_min' for col in minimum.columns]





        

mean= features.groupby(['series_id']).mean()

mean.columns = [str(col) + '_mean' for col in mean.columns]



median= features.groupby(['series_id']).median()

median.columns = [str(col) + '_median' for col in median.columns]

mean=mean.join(median,lsuffix='series_id',rsuffix='series_id')







mean=mean.join(minimum,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(maximum,lsuffix='series_id',rsuffix='series_id')



std=features.groupby(['series_id']).std()

std.columns = [str(col) + '_std' for col in std.columns]

mean=mean.join(std,lsuffix='series_id',rsuffix='series_id')



skew=features.groupby(['series_id']).skew()

skew.columns = [str(col) + '_skew' for col in skew.columns]

mean=mean.join(skew,lsuffix='series_id',rsuffix='series_id')



rangeV=pd.DataFrame()

maxToMin=pd.DataFrame()

mean_abs_chg=pd.DataFrame()

mean_change_of_abs_change=pd.DataFrame()

abs_max=pd.DataFrame()

abs_min=pd.DataFrame()

abs_avg=pd.DataFrame()



for col in features.columns:

    if col=='series_id':

        pass

    else:

        rangeV[col + '_range'] = maximum[col + '_max'] - minimum[col + '_min']

        maxToMin[col + '_maxtoMin'] = maximum[col + '_max'] / minimum[col + '_min']

        mean_abs_chg[col + '_mean_abs_chg'] = features.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        mean_change_of_abs_change[col + '_mean_change_of_abs_change'] = features.groupby('series_id')[col].apply(lambda x: np.mean(np.diff(np.abs(np.diff(x)))))

        abs_max[col + '_abs_max'] = features.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        abs_min[col + '_abs_min'] = features.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        abs_avg[col + '_abs_avg'] = (abs_min[col + '_abs_min'] + abs_max[col + '_abs_max'])/2

mean=mean.join(rangeV,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(maxToMin,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(mean_abs_chg,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(mean_change_of_abs_change,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(abs_max,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(abs_min,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(abs_avg,lsuffix='series_id',rsuffix='series_id')







features_test=mean


features=X_train

features=features.drop(features.columns[0], axis=1)

features=features.drop(features.columns[0],axis=1)

features=features.drop(features.columns[1],axis=1)







maximum=features.groupby(['series_id']).max()

maximum.columns = [str(col) + '_max' for col in maximum.columns]





minimum=features.groupby(['series_id']).min()

minimum.columns = [str(col) + '_min' for col in minimum.columns]





        

mean= features.groupby(['series_id']).mean()

mean.columns = [str(col) + '_mean' for col in mean.columns]



median= features.groupby(['series_id']).median()

median.columns = [str(col) + '_median' for col in median.columns]

mean=mean.join(median,lsuffix='series_id',rsuffix='series_id')







mean=mean.join(minimum,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(maximum,lsuffix='series_id',rsuffix='series_id')



std=features.groupby(['series_id']).std()

std.columns = [str(col) + '_std' for col in std.columns]

mean=mean.join(std,lsuffix='series_id',rsuffix='series_id')



skew=features.groupby(['series_id']).skew()

skew.columns = [str(col) + '_skew' for col in skew.columns]

mean=mean.join(skew,lsuffix='series_id',rsuffix='series_id')



rangeV=pd.DataFrame()

maxToMin=pd.DataFrame()

mean_abs_chg=pd.DataFrame()

mean_change_of_abs_change=pd.DataFrame()

abs_max=pd.DataFrame()

abs_min=pd.DataFrame()

abs_avg=pd.DataFrame()



for col in features.columns:

    if col=='series_id':

        pass

    else:

        rangeV[col + '_range'] = maximum[col + '_max'] - minimum[col + '_min']

        maxToMin[col + '_maxtoMin'] = maximum[col + '_max'] / minimum[col + '_min']

        mean_abs_chg[col + '_mean_abs_chg'] = features.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        mean_change_of_abs_change[col + '_mean_change_of_abs_change'] = features.groupby('series_id')[col].apply(lambda x: np.mean(np.diff(np.abs(np.diff(x)))))

        abs_max[col + '_abs_max'] = features.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        abs_min[col + '_abs_min'] = features.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        abs_avg[col + '_abs_avg'] = (abs_min[col + '_abs_min'] + abs_max[col + '_abs_max'])/2

mean=mean.join(rangeV,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(maxToMin,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(mean_abs_chg,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(mean_change_of_abs_change,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(abs_max,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(abs_min,lsuffix='series_id',rsuffix='series_id')

mean=mean.join(abs_avg,lsuffix='series_id',rsuffix='series_id')







features=mean


clf1 = RandomForestClassifier(max_depth=12)

scores = cross_val_score(clf1, features, y_train['surface'],cv=5)

avg=((scores[0]+scores[1]+scores[2]+scores[3]+scores[4])/5)

print('CV AVG:'+str(avg))

results={}



clf=RandomizedSearchCV(SVC(),dict(C=np.logspace(.001,1,num=100), gamma=np.logspace(.001,1,num=100), degree=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]),n_iter=5,random_state=0)

results = clf.fit(features, y_train['surface'])

print(results.best_params_)
clf = SVC(gamma=1.0023, degree=6, C=1.750)

scores = cross_val_score(clf, features, y_train['surface'],cv=5)

avg=((scores[0]+scores[1]+scores[2]+scores[3]+scores[4])/5)

print('CV AVG:'+str(avg))
clf1.fit(features, y_train['surface'])

y_pred=clf1.predict(features_test)

print('Accuracy: '+str(accuracy_score(y_test['surface'], y_pred)))
clf.fit(features, y_train['surface'])

y_pred=clf.predict(features_test)

print('Accuracy: '+str(accuracy_score(y_test['surface'], y_pred)))