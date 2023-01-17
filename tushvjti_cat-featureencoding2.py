# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVR

from sklearn import *

import datetime as dt
#### Import Dependencies

%matplotlib inline

#### Start Python Imports

import math, time, random, datetime

#### Data Manipulation

import numpy as np

import pandas as pd

#### Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



#### Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



#### Machine learning

import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier, Pool, cv



##### Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')

print ("Data is loaded!")
def RMSLE(y, pred):

    return metrics.mean_squared_error(y, pred) ** 0.5
data = train.copy()

valid = test.copy()
#data.nunique()

#valid.nunique()



# in case needs
# get a list of object cat columns 

# Get list of categorical variables

s = (data.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
len (object_cols)

OH_col = data.loc[:, data.nunique() < 15].columns



new_OH = []

for x in OH_col:

    if x in object_cols:

        new_OH.append(x)

        

#new_OH
LE_col = data.loc[:, data.nunique() >= 15].columns

new_LE = []

for x in LE_col:

    if x in object_cols:

        new_LE.append(x)



#new_LE
# Make copy to avoid changing original data 

label_X_train = data.copy()

label_X_valid = valid.copy()
# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in new_LE:

    label_X_train[col] = label_encoder.fit_transform(data[col])

    label_X_valid[col] = label_encoder.fit_transform(valid[col])

print(label_X_train.shape)

print(label_X_valid.shape)
label_X_train.head(2)
label_X_valid.head(2)
# use label_X_train and label_X_valid for next calculations ( One hot encoding )
#label_X_train[new_OH].nunique()
# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(label_X_train[new_OH]))

OH_cols_valid = pd.DataFrame(OH_encoder.fit_transform(label_X_valid[new_OH]))

##  check if fit_transform or just transform should be used.... for valid data set....

print(OH_cols_train.shape)

print(OH_cols_valid.shape)
label_X_train[new_OH].nunique().sum()

# means OH_cols_train has no data of rest of columns....

# so now add the data back
# One-hot encoding removed index; put it back

OH_cols_train.index = label_X_train.index

OH_cols_valid.index = label_X_valid.index
# Remove categorical columns (will replace with one-hot encoding)

# these are columns which has numerical data and lebel encoding columns that's been processed already.

num_X_train = label_X_train.drop(new_OH, axis=1)

num_X_valid = label_X_valid.drop(new_OH, axis=1)
#num_X_train.head(2)
#num_X_valid.head(2)
# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
#OH_X_train.head(2)
#OH_X_valid.head(2)
print(OH_X_train.shape)

print(OH_X_valid.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.utils.testing import ignore_warnings
rf = RandomForestClassifier( 

                             n_estimators=200,

                             n_jobs=-1,

                             verbose = 2)

#model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)



lr1 = LogisticRegression(solver='lbfgs', C=0.1)
X_train = OH_X_train.drop("target", axis = 1)

y_train = OH_X_train["target"]

X_train = X_train.drop("id", axis=1)

X_test = OH_X_valid.drop("id",axis = 1)
#scaler = MinMaxScaler(feature_range=(0, 1))

#X_train = scaler.fit_transform(X_train)

#X_test = scaler.fit_transform(X_test)
print(X_train.shape)

print(y_train.shape)
#rf.fit(X_train, y_train)

#lr1.fit(X_train, y_train)
# alternate cv method

X, X_hideout, y, y_hideout = model_selection.train_test_split(X_train, y_train, test_size=0.13, random_state=42)
# Set up folds

K = 4

kf = model_selection.KFold(n_splits = K, random_state = 1, shuffle = True)

np.random.seed(1)
#model = SVR(kernel='rbf')

params = {'n_estimators': 10, # change to 9000 to obtain 0.505 on LB (longer run time expected)

        'max_depth': 5,

        'min_samples_split': 200,

        'min_samples_leaf': 50,

        'learning_rate': 0.005,

        'max_features':  'sqrt',

        'subsample': 0.8,

        'loss': 'ls'}

#model = ensemble.GradientBoostingRegressor(**params)

model = ensemble.RandomForestClassifier(n_jobs = -1, verbose = 2)
print("Started CV at ", dt.datetime.now())

for i, (train_index, test_index) in enumerate(kf.split(X)):

    # Create data for this fold

    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]

    X_train, X_valid = X.iloc[train_index, :].copy(), X.iloc[test_index, :].copy()

    #X_test = test[col]

    print("\nFold ", i)

    

    fit_model = model.fit(X_train, y_train)

    pred = model.predict(X_valid)

    print('RMSLE GBM Regressor, validation set, fold ', i, ': ', RMSLE(y_valid, pred))

    

    pred_hideout = model.predict(X_hideout)

    print('RMSLE GBM Regressor, hideout set, fold ', i, ': ', RMSLE(y_hideout, pred_hideout))

    print('Prediction length on validation set, GBM Regressor, fold ', i, ': ', len(pred))

    # Accumulate test set predictions

    

    del  X_train, X_valid, y_train

    

print("Finished CV at ", dt.datetime.now())
# scores = []

# best_svr = SVR(kernel='rbf')

# #random_state=42, shuffle=False

# cv = KFold(n_splits=10)

# for train_index, test_index in cv.split(X_train):

#     print("Train Index: ", train_index, "\n")

#     print("Test Index: ", test_index)



#     X_tr = X_train.iloc[train_index,:]

#     X_tes = X_train.iloc[test_index,:]

#     y_tr = y_train.iloc[train_index]

#     y_tes = y_train.iloc[test_index]

#     print(X_tr.shape)

#     print(X_tes.shape)

#     print(y_tr.shape)

#     print(y_tes.shape)

    

    

#     #best_svr.fit(X_tr, y_tr)

#     #scores.append(best_svr.score(X_tes, y_tes))
#X_train.iloc[[1,3],:]

#y_train.iloc[30000]
X_test.head(2)
# predictions = rf.predict(X_test)

# predict_lr = lr1.predict_proba(X_test)

# prediction_svr = best_svr.predict(X_test)



# submission = pd.DataFrame()

# submission_LR = pd.DataFrame()

# submission_svr = pd.DataFrame()



# submission["id"] = OH_X_valid["id"]

# submission_LR["id"] = OH_X_valid["id"]

# submission_svr['id'] = OH_X_valid["id"]



# submission["target"] = predictions

# submission_LR["target"] = predict_lr[:, 1]

# submission_svr["target"] = prediction_svr
prediction = model.predict(X_test)

submission = pd.DataFrame()

submission['id'] = OH_X_valid["id"]

submission["target"] = prediction
submission.to_csv("cat_submission1.csv", index = False)
predict_lr[:, 1]
submission.target.value_counts().sum()
submission.to_csv("cat_submission1.csv", index = False)

submission_LR.to_csv("cat_submission_lr.csv", index = False)
from sklearn.model_selection import cross_validate



score=cross_validate(lr1, X_train, y_train, cv=3, scoring="roc_auc")["test_score"].mean()

print(f"{score:.6f}")