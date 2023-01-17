# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in Expand all 10 unchanged lines

# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd

train =  pd.read_table('../input/train.tsv')

drpNa = train.drop(['train_id','name','category_name',"brand_name","item_description"],1)

drpNa = drpNa.dropna() 



def rmsle(h, y): 

    """

    Compute the Root Mean Squared Log Error for hypthesis h and targets y

     Args:

             h - numpy array containing predictions with shape (n_samples, n_targets)

             y - numpy array containing targets with shape (n_samples, n_targets)

    """

    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

print(train.head())

from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split( 

drpNa[['item_condition_id','shipping']][:10000],drpNa['price'][:10000] , test_size= 0.4 , random_state= 0 ) 

from sklearn.svm import SVR



clf = SVR(C=1.0,epsilon=0.2)

clf.fit(X_train,y_train)

pre = clf.predict(X_test)

rmsle(pre,y_test)



test =  pd.read_table('../input/test.tsv')



TdrpNa = test.drop(['test_id','name','category_name',"brand_name","item_description"],1)

TdrpNa = TdrpNa.dropna() 

trial_sub1 = clf.predict(TdrpNa)



submission  = pd.read_csv('../input/sample_submission.csv')

submission['price']=trial_sub1

submission.to_csv('fist_trial.csv',index=False)



len(submission)