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
train = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv',index_col = 0)

test = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv',index_col=0)

submit = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/sampleSubmission.csv',index_col=0)
del train['player_positions']

del train['work_rate']

del train['joined']

del train['nation_position']

del train['preferred_foot']

del train['nation_jersey_number']

del train['player_traits']

del train['team_position']

#train = train.dropna(subset=['team_position'])



del test['player_positions']

del test['work_rate']

del test['joined']

del test['nation_position']

del test['preferred_foot']

del test['nation_jersey_number']

del test['player_traits']

del test['team_position']

train=train.fillna({'pace':0,'shooting':0,'passing':0,'dribbling':0,'defending':0,'physic':0,'gk_diving':0,'gk_handling':0,'gk_kicking':0,'gk_reflexes':0,'gk_speed':0,'gk_positioning':0})

test = test.fillna({'pace':0,'shooting':0,'passing':0,'dribbling':0,'defending':0,'physic':0,'gk_diving':0,'gk_handling':0,'gk_kicking':0,'gk_reflexes':0,'gk_speed':0,'gk_positioning':0})
print(train.info())
train
for col in ['ls','st','rs','lw','lf','cf','rf','rw','lam','cam','lm','lcm','cm','ram','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb']:

    a=train[col]

    for sum in range(12795):

        if a[sum] is not np.NaN:

            a[sum] = eval(a[sum])

        else :

            a[sum] = 0

    train[col]=a

    train[col]=train[col].astype(int)

train = train.fillna(train.mean())

#team_jersey_number

#contract_valid_until 
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

train['loaned']=le.fit_transform(train['loaned'])

train['loaned']=train['loaned'].astype(int)
print(test.info()) #5483
test
for col in ['ls','st','rs','lw','lf','cf','rf','rw','lam','cam','lm','lcm','cm','ram','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb']:

    b=test[col]

    for num in range(12795,18278):

        if b[num] is not np.NaN:

            b[num] = eval(b[num])

        else :

            b[num] = 0

    test[col]=b

    test[col]=test[col].astype(float)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

test['loaned']=le.fit_transform(test['loaned'])

test['loaned']=test['loaned'].astype(int)
test = test.fillna(test.mean())
X = train.drop('value_eur',axis=1).values

y = train['value_eur'].values
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X,y)

model.score(X,y)
X_test = test.values

p = model.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/sampleSubmission.csv',index_col=0)

submit_df['value_eur'] = p

submit_df.to_csv('submission.csv')