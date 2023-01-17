# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/novartis/Train.csv')

train.head(5)
import datetime

#train['month'] = pd.DatetimeIndex(train['DATE']).month

train['year']  = pd.DatetimeIndex(train['DATE']).year
train['MULTIPLE_OFFENSE1']=train['MULTIPLE_OFFENSE']
train = train.drop(['MULTIPLE_OFFENSE'],axis=1)
train.head(4)
X = train.iloc[: ,2:-1].values

y = train.iloc[: , -1].values

X
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

imputer = imputer.fit(X)

X = imputer.transform(X)
y
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(random_state=1,

                             n_estimators=190,

                             criterion='gini',

                            max_depth=25,

                             min_samples_split=2,

                             min_samples_leaf=1,

                             bootstrap=True,

                            n_jobs=-1,verbose=False)



clf.fit(X,y)
from xgboost import XGBClassifier

classifier_xgb = XGBClassifier()

classifier_xgb.fit(X,y)

#clf.score(X_test,y_test)
test = pd.read_csv('/kaggle/input/novartis/Test.csv')

test.head()
import datetime

#test['month'] = pd.DatetimeIndex(test['DATE']).month

test['year']  = pd.DatetimeIndex(test['DATE']).year
test1 = test.iloc[:,2:]

test1.head()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

imputer = imputer.fit(test1)

test1 = imputer.transform(test1)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

test1 = sc_X.fit_transform(test1)
prediction = classifier_xgb.predict(test1)

prediction
test.head(2)
submission = pd.DataFrame({'INCIDENT_ID':test['INCIDENT_ID'],'MULTIPLE_OFFENSE':prediction})

submission.head()
filename = 'Novartis_submission_21.csv'

submission.to_csv(filename, index = False)

print('saved file:',filename)