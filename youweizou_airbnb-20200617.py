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
import datetime

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv("/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip")

test = pd.read_csv("/kaggle/input/airbnb-recruiting-new-user-bookings/test_users.csv.zip")

train.shape, test.shape
train.head()
# Before doing anything, let's get all the dates to date type

dataset = pd.concat([train, test])

dataset['date_account_created'] = pd.to_datetime(dataset['date_account_created'])

dataset['timestamp_first_active'] = pd.to_datetime(dataset['timestamp_first_active'].astype('object'), format='%Y%m%d%H%M%S')

dataset['date_first_booking'] = pd.to_datetime(dataset['date_first_booking'])

dataset.head()
sns.distplot(dataset['age'])
dataset['signup_flow'] = dataset['signup_flow'].astype('object')

obj = []

for col in dataset.drop(columns='id').columns:

    if dataset[col].dtype == 'object':

        obj.append(col)

len(obj)
plt.figure(figsize=(12, 18))

for i, c in enumerate(obj):

    plt.subplot(4,3,i+1)

    sns.countplot(dataset[c])

    plt.xticks(rotation=60)

plt.tight_layout()    

plt.show()
missing_data = dataset.isnull().sum().sort_values(ascending=False)

missing_percent = missing_data/len(dataset)

pd.DataFrame({'Count': missing_data, 'Percent': missing_percent})
# Too many missing values in "date_first_booking"

dataset = dataset.drop(columns= ['date_first_booking', 'id'])
# Seperate year, month, day from date columns



dataset['year_account_created'] = pd.DatetimeIndex(dataset['date_account_created']).year

dataset['month_account_created'] = pd.DatetimeIndex(dataset['date_account_created']).month

dataset['day_account_created'] = pd.DatetimeIndex(dataset['date_account_created']).day

dataset['year_first_active'] = pd.DatetimeIndex(dataset['timestamp_first_active']).year

dataset['month_first_active'] = pd.DatetimeIndex(dataset['timestamp_first_active']).month

dataset['day_first_active'] = pd.DatetimeIndex(dataset['timestamp_first_active']).day

dataset['hour_first_active'] = pd.DatetimeIndex(dataset['timestamp_first_active']).hour

dataset['minute_first_active'] = pd.DatetimeIndex(dataset['timestamp_first_active']).minute

# dataset['year_first_booking'] = pd.DatetimeIndex(dataset['date_first_booking']).year

# dataset['month_first_booking'] = pd.DatetimeIndex(dataset['date_first_booking']).month

# dataset['day_first_booking'] = pd.DatetimeIndex(dataset['date_first_booking']).day

dataset = dataset.drop(columns=['date_account_created', 'timestamp_first_active'])
# "country destination" is the target, the missing data is all from test 

# only age and first_affiliate_tracked need to be addressed(as well as the info seperated from "first booking")

# before dealing with age, remember we found that there were some outliers so let's remove/modify the outliers first
age_values = dataset['age'].values

dataset['age'] = np.where(np.logical_or(age_values>14, age_values<90), age_values, -1)

dataset['first_affiliate_tracked'] = dataset['first_affiliate_tracked'].fillna(dataset['first_affiliate_tracked'].mode()[0])

# dataset.isnull().sum().sum()
dataset.head()
data_dummy = pd.get_dummies(dataset.drop(columns='country_destination'))
df_train = data_dummy[:len(train)]

y = train['country_destination']

df_test = data_dummy[len(train):]

df_train.head()
le = LabelEncoder()

y = le.fit_transform(y)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from xgboost import XGBClassifier, plot_importance

from lightgbm import LGBMClassifier

from sklearn.model_selection import learning_curve

import xgboost as xgb
dtrain = xgb.DMatrix(df_train, y)

params = {

    'objective': 'multi:softprob',              # So beneficial to multi classifications

    'max_depth': 6,

    'eval_metric': 'merror',                    # Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases)

    'learning_rate': 0.3,                       # https://xgboost.readthedocs.io/en/latest/parameter.html this link includes all the params and the options

    'colsample_bytree': 0.3,                    # of each params

    'subsample': 0.5,

    'num_class': len(pd.Series(y).value_counts().index)

}

res = xgb.cv(params, dtrain, num_boost_round=50, nfold=5, early_stopping_rounds=1, verbose_eval=1, show_stdv=True)
clf = xgb.train(dtrain=dtrain, params=params, num_boost_round=res['test-merror-mean'].idxmin())
plot_importance(clf, max_num_features=10)
dtest = xgb.DMatrix(df_test)

y_pred = clf.predict(dtest)

ids = []

pred = []

for i in range(len(test)):

    ids += [test['id'][i]]*5

    pred += list(le.inverse_transform(y_pred[i].argsort()[::-1][:5]))

sub = pd.DataFrame({'id': ids, 'country': pred})

sub.to_csv('sub.csv', index=False)