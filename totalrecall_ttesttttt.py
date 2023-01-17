# Imports



# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn, sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
sns.set_style('whitegrid')
##%matplotlib inline



#### import the data
train_users   = pd.read_csv('../input/train_users_2.csv')
test_users    = pd.read_csv('../input/test_users.csv')
gender = pd.read_csv('../input/age_gender_bkts.csv')
sessions = pd.read_csv('../input/sessions.csv')
countries = pd.read_csv('../input/countries.csv')

countries
countries_trimmed = countries[['country_destination', 'language_levenshtein_distance', 'distance_km']]
countries_trimmed.transpose()
countries_trimmed
languages_trimmed = countries[['destination_language ', 'language_levenshtein_distance']]
#languages_trimmed = languages_trimmed.drop_duplicates(cols = 'destination_language ', inplace = True)
languages_trimmed.head()


test_users[test_users['id'] == 'ailzdefy6o']
train_users[train_users['id'].str.contains('s7o398rfc5', na=False)]
test_users[test_users['id'].str.contains('gnwswj64nx', na=False)]
sessions[sessions['user_id'].str.contains('gnwswj64nx', na=False)]
##np.unique([sessions['action']])
sessions[sessions['action'].str.contains('translate', na=False)]
sessions.head()

train_users[train_users['language'].str.contains('fra', na=False)]
Series(train_users['language'].ravel()).unique()
train_users.head()
for col in train_users:
    print(col)
import sys
import numpy as np
import pandas as pd

sessions= pd.read_csv('../input/sessions.csv')

print(sessions.head())
print(sessions.info())
print(sessions.apply(lambda x: x.nunique(),axis=0))

#sessions['action'] = sessions['action'].fillna('999')
#data roll-up
#secs_elapsed
grpby = sessions.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
grpby.columns = ['user_id','secs_elapsed']

# agg = grpby['secs_elapsed'].agg({'time_spent' : np.sum})

#action
#print(sessions.action_type.value_counts())
#print(sessions.groupby(['action_type'])['user_id'].nunique().reset_index())
action_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['action_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
action_type = action_type.drop(['booking_response'],axis=1)
#print(action_type.head())

#print(sessions.groupby(['device_type'])['user_id'].nunique().reset_index())
#print(sessions.groupby(['user_id'])['device_type'].nunique().reset_index())
device_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['device_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
device_type = device_type.drop(['Blackberry','Opera Phone','iPodtouch','Windows Phone'],axis=1)
#device_type = device_type.replace(device_type.iloc[:,1:]>0,1)
print(device_type.info())

sessions_data = pd.merge(action_type,device_type,on='user_id',how='inner')

sessions_data = pd.merge(sessions_data,grpby,on='user_id',how='inner')
test = pd.merge(sessions_data, test_users, left_on = 'user_id', right_on = 'id')
test
