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

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')

test = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/test_users.csv.zip')

age_gender = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/age_gender_bkts.csv.zip')

sessions = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv.zip')
train.info()
train.head()
X_train = train.drop(['date_first_booking', 'country_destination'], axis=1)

X_test = test.drop(['date_first_booking'], axis=1)
y_des = train['country_destination'].values

X=pd.concat((X_train, X_test), axis=0, ignore_index=True)

X.shape
X.fillna(method='pad').head()
X.loc[X.age > 90, 'age'] = -1

X.loc[X.age < 13, 'age'] = -1

X['age'].describe()
X.loc[X.age.isnull(), 'age']=X.age.mean()
dac = np.vstack(

    X.date_account_created.astype(str).apply(

        lambda x: list(map(int, x.split('-')))

    ).values

)

X['dac_year'] = dac[:, 0]

X['dac_month'] = dac[:, 1]

X['dac_day'] = dac[:, 2]

X = X.drop(['date_account_created'], axis=1)

X.head()
df = sessions.user_id.value_counts()

print(df.shape)

print(df.head())
df = df.to_frame()

df = df.rename(columns = {'user_id' : 'session_count'})

df['id'] = df.index

df.head()
X = pd.merge(X, df, how = 'left', on = ['id'])

X.session_count.fillna(-1, inplace = True)

X.session_count = X.session_count.astype(int)
tfa = np.vstack(

    X.timestamp_first_active.astype(str).apply(

        lambda x: list(map(int, [x[:4], x[4:6], x[6:8],

                                 x[8:10], x[10:12],

                                 x[12:14]]))

    ).values

)

X['tfa_year'] = tfa[:, 0]

X['tfa_month'] = tfa[:, 1]

X['tfa_day'] = tfa[:, 2]

X = X.drop(['timestamp_first_active'], axis=1)
# age distributions

train['corrected_age']=train['age'].apply(lambda x : 36 if x>90 or x<10 else x)

sns.distplot(train.corrected_age.dropna())
# percentage of users using different signup_method

signup_method = X.signup_method.value_counts(dropna = False) / len(X) * 100

signup_method.plot('bar', rot = 0)

plt.xlabel('Sign up method')

plt.ylabel('Percentage of signup_method')
# percentage of gender

gender = X.gender.value_counts(dropna = False) / len(X) * 100

gender.plot('bar', rot = 0)

plt.xlabel('gender')

plt.ylabel('Percentage of gender')
# percentage of people going to different countries

des_countries = train.country_destination.value_counts(dropna = False) / len(train) * 100

des_countries.plot('bar', rot = 0)

plt.xlabel('Destination country')

plt.ylabel('Percentage of booking')
# Relavance between Age and destination

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(10, 7)

sns.boxplot(y='age' , x='country_destination',data=train)

plt.xlabel('Destination Country box plot',size=15)

plt.ylabel('Age of Users', size=15)

plt.tick_params(labelsize=12)
# relevance between age and signup method

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(6, 4)

sns.boxplot(y='age' , x='signup_method',data=train)

plt.xlabel('Signup method', size=15)

plt.ylabel('age', size=15)

plt.tick_params(labelsize=12)

#sns.despine()
# relevence between age and signup app

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(6, 4)

sns.boxplot(y='age' , x='signup_app',data=train)

plt.xlabel('Signup app',size=15)

plt.ylabel('Age of Users', size=15)

plt.tick_params(labelsize=12)

#sns.despine()
#relevence between age and language

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(8, 5)

sns.boxplot(y='age' , x='language',data=train)

plt.xlabel('Language', size=15)

plt.ylabel('Age of Users', size=15)

plt.tick_params(labelsize=12)

#sns.despine()
# relevance between age and gender

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(6, 4)

sns.boxplot(y='age' , x='gender',data=train)

plt.xlabel('Gender', size=15)

plt.ylabel('Age of Users', size=15)

plt.tick_params(labelsize=10)

#sns.despine()
# chart for number of account created

train['date_account_created_new'] = pd.to_datetime(train['date_account_created'])

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(10, 8)

train.date_account_created_new.value_counts().plot(kind='line', linewidth=1, color='#1F618D')

plt.xlabel('Date ', size=20)

plt.ylabel('Number of account created ', size=15)

plt.tick_params(labelsize=12)

#sns.despine()
oh_features = ['gender', 'signup_method', 'signup_flow', 'language',

                'affiliate_channel', 'affiliate_provider',

                'first_affiliate_tracked', 'signup_app',

                'first_device_type', 'first_browser']
for feature in oh_features:

    X_dummy = pd.get_dummies(X[feature], prefix=feature)

    X = X.drop([feature], axis=1)

    X = pd.concat((X, X_dummy), axis=1)

X.head()
#split the well processed dataset into X_train and X_test

X_train = X.iloc[:len(train), :]

X_test = X.iloc[len(train):, :]

X_train = X_train.drop(['id'], axis=1)

X_train.shape

X_test = X_test.drop(['id'], axis=1)
le = LabelEncoder()

y_trans = le.fit_transform(y_des)

y_trans.shape
dtrain, dtest, train_label, test_label = train_test_split(X_train, y_trans, test_size = 0.3, random_state = 817)
#logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(dtrain, train_label)

pred_log=logreg.predict(dtest)

from sklearn.metrics import accuracy_score

print(accuracy_score(test_label, pred_log))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=20, n_estimators=100)

rfc.fit(dtrain , train_label)

pred = rfc.predict(dtest)

print(accuracy_score(test_label, pred))
fi=pd.Series(rfc.feature_importances_, index=dtrain.columns)

fn=fi.sort_values(ascending=True)

fn[-20:].plot(kind='barh', color='r', figsize=(25, 12))

plt.xlabel('importance', size=15)

plt.title('Random Forest Importance', size=20)

plt.tick_params(labelsize=15)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=10)

dtc.fit(dtrain , train_label)

pred = dtc.predict(dtest)

print(accuracy_score(test_label, pred))
from xgboost.sklearn import XGBClassifier

xgb = XGBClassifier(max_depth=4, learning_rate=0.03, n_estimators=100,

                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=40)

xgb.fit(dtrain , train_label)

pred = xgb.predict(dtest) 

print(accuracy_score(test_label, pred))
# only XGBoost

xgb = XGBClassifier(max_depth=4, learning_rate=0.03, n_estimators=100,

                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=40)

xgb.fit(X_train, y_trans)

XGBC_pred_test = xgb.predict(X_test)

XGBC_pred_test_prob=xgb.predict_proba(X_test)
ids_test = test['id']



ids = []

countries = []



for i in range(len(X_test)):

    idx = ids_test[i]

    ids += [idx] * 5

    countries += le.inverse_transform(np.argsort(XGBC_pred_test_prob[i])[::-1][:5]).tolist()
submission = pd.DataFrame({

    "id" : ids,

    "country" : countries

})

submission.to_csv('submission_XGBC.csv', index = False)
n_labels=len(set(y_des))

n_labels
params = {

    'objective': 'multi:softprob',

    'eval_metric': 'merror',

    'num_class': n_labels,

    'eta': 0.5,

    'max_depth': 6,

    'subsample': 0.5,

    'colsample_bytree': 0.3,

    'silent': 1,

    'seed': 123

}
import xgboost as xgb

num_boost_round = 50



Dtrain = xgb.DMatrix(X_train, y_trans)

res = xgb.cv(params, Dtrain, num_boost_round=num_boost_round, nfold=5,

             callbacks=[xgb.callback.print_evaluation(show_stdv=True),

                        xgb.callback.early_stop(50)])
num_boost_round = res['test-merror-mean'].idxmin()

print(format(num_boost_round))

clf = xgb.train(params, Dtrain, num_boost_round=num_boost_round)

clf
import operator

importance = clf.get_fscore()

importance_df = pd.DataFrame(

    sorted(importance.items(), key=operator.itemgetter(1)),

    columns=['feature', 'fscore']

)
importance_df = importance_df.iloc[-20:, :]
plt.figure()

importance_df.plot(kind='barh', x='feature', y='fscore',

                   legend=False, figsize=(20, 10))

plt.title('XGBoost Feature Importance', size=25)

plt.xlabel('Relative importance', size=20)

plt.ylabel('Features', size=20)

plt.tick_params(labelsize=15)

#plt.gcf().savefig('feature_importance.png')