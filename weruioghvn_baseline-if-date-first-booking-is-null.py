import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
print(os.listdir('../input'))
file = ['train_users.csv', 'age_gender_bkts.csv', 'sessions.csv', 'countries.csv', 'test_users.csv']
data = {}
for f in file:
    data[f.replace('.csv','')]=pd.read_csv('../input/'+f)
    
train = data['train_users']
test = data['test_users']
# train = train.fillna(-100)
# test = test.fillna(-100)

age = data['age_gender_bkts']
sessions = data['sessions']
country = data['countries']
target = train['country_destination']
train = train.drop(['country_destination'],axis=1)


result = []
for index, row in test.iterrows():
    if isinstance(row['date_first_booking'], float):
        result.append([row['id'], 'NDF'])
        result.append([row['id'], 'US'])
        result.append([row['id'], 'other'])
        result.append([row['id'], 'FR'])
        result.append([row['id'], 'IT'])
    else:
        result.append([row['id'], 'US'])
        result.append([row['id'], 'other'])
        result.append([row['id'], 'FR'])
        result.append([row['id'], 'IT'])
        result.append([row['id'], 'GB'])
        
pd.DataFrame(result).to_csv('sub.csv', index = False, header = ['id', 'country'])
# temp = pd.DataFrame(train.apply(lambda row: isinstance(row['date_first_booking'], float), axis = 1))
# temp['destination'] = (target == 'NDF')
# temp['comparison'] = temp.apply(lambda x: x[0] != x['destination'], axis = 1)
# temp.apply(sum)