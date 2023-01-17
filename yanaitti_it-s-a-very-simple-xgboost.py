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
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import metrics, preprocessing

from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer

train = pd.read_csv('/kaggle/input/titanic/train.csv', index_col=0)

test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col=0)

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train.shape, test.shape, submission.shape, train, test, submission
train.info()
cols_o = train.select_dtypes(include='object').columns.tolist()

cols_o
test['Survived'] = -999
all_df = pd.concat([train, test], axis=0)

all_df
all_df.isnull().sum()
all_df['Cabin'].unique()
all_df['Cabin'] = all_df['Cabin'].fillna('unknown')

all_df['Cabin']
all_df['Cabin'] = all_df['Cabin'].apply(lambda x: x if len(x.split(' ')) == 1 else 'unknown')

all_df['Cabin'].unique()
all_df['Embarked'].unique()
all_df['Embarked'] = all_df['Embarked'].fillna('NA')

all_df['Embarked'].unique()
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_data = imp_mean.fit_transform(all_df[['Fare', 'Age']].values)

all_df['Fare'] = imp_data[:,0]

all_df['Age'] = imp_data[:,1]

all_df
all_df.isnull().sum()
all_df
for column in cols_o:

    le = preprocessing.LabelEncoder()

    le.fit(all_df[column])

    all_df[column] = le.transform(all_df[column])



all_df
X = all_df[all_df['Survived'] != -999].drop(['Survived', 'Name', 'Ticket'], axis=1)

y = all_df[all_df['Survived'] != -999]['Survived']



X_Test = all_df[all_df['Survived'] == -999].drop(['Survived', 'Name', 'Ticket'], axis=1)



X.shape, y.shape, X_Test.shape
X
params = {

    'learning_rate': [0.1, 0.01, 0.001],

    'max_depth': list(range(1, 11)),

    'subsample': [0.1, 0.3, 0.5, 0.7, 0.9],

    'n_estimators': [10, 100, 1000],

}

xgb = XGBClassifier()



gcv = GridSearchCV(xgb, params, cv=5, return_train_score=True)

gcv.fit(X, y)
gcv.best_params_
gcv.cv_results_
train_score = gcv.cv_results_['mean_train_score']

test_score = gcv.cv_results_['mean_test_score']
gcv.cv_results_
plt.figure(figsize=(20, 4))

plt.plot(train_score)

plt.plot(test_score)

plt.xticks(list(range(0, 10)), list(range(0, 450)))

pred = gcv.predict(X_Test)

pred
submission['Survived'] = pred
submission.to_csv('submit_20200815_002.csv', index=None)