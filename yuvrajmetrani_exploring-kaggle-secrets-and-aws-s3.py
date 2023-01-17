import numpy as np

from dateutil.relativedelta import relativedelta

import pandas as pd

import boto3

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

bucket = user_secrets.get_secret("bucket")

file_name = user_secrets.get_secret("file_name")

key_id = user_secrets.get_secret("key_id")

secret_key = user_secrets.get_secret("secret_key")



s3 = boto3.client('s3', aws_access_key_id=key_id, aws_secret_access_key=secret_key) 





# get object and file (key) from bucket

obj = s3.get_object(Bucket= bucket, Key= file_name) 



#read the content body

df = pd.read_csv(obj['Body'], parse_dates=['date_of_birth'], infer_datetime_format=True)

df
df.info()
df['date_of_birth'].max()
df['date_of_birth'].min()
y = relativedelta(df['date_of_birth'].max(), df['date_of_birth'].min()).years

print(f'The age gap between the oldest and the youngest record is {y} years.')
df.count()
(1 - 387/500)

#try:



#df['age'] = df.apply(lambda x: relativedelta(today, x['date_of_birth']).years)

#except:

#    print('check outcome. lambda not successful')

today = pd.to_datetime('today')

df['age'] = df['date_of_birth']

for row in range(0,499):

    df['age'][row] = (relativedelta(today, df['date_of_birth'][row]).years)
df
df2 = pd.DataFrame(df.device_type_us_state.str.split('-',1).tolist(),columns = ['device_type','us_state'])



df = pd.concat([df,df2], axis=1, sort=False)

df
df['Gender'] = df['Gender'].apply(lambda x: 'M' if x == 'MALE' else x)

df['Gender'] = df['Gender'].apply(lambda x: 'M' if x == 'Male' else x)

df['Gender'] = df['Gender'].apply(lambda x: 'F' if x == 'FEMALE' else x)

df['Gender'] = df['Gender'].apply(lambda x: 'F' if x == 'Female' else x)

df


df_dropped_invalid_age.Gender.value_counts()
df['us_state'].nunique()
df.us_state.value_counts()
df.device_type.value_counts()
df.age.value_counts()
df_bad_age_dropped = df.drop(499)
df_bad_age_dropped
import numpy as np

age_groups = pd.cut(df_bad_age_dropped['age'], bins=[0,18, 24, 34,44, 54, 64, 100], right=True, precision=3)

#df.groupby(age_groups).count()

age_groups

# Age Groups = 0-18, 19-24,25-34,35-44, 45-54, 55-64,65+ 
df_age_grouped = (df_bad_age_dropped.groupby(age_groups).count())

#df_age_grouped.set_index()

df_age_grouped.columns
df_age_grouped
df_age_grouped['age']
plt.figure(figsize=(14,7))

plt.xlabel("age")

plt.ylabel("value_counts")

plt.title("Mobile Devices against Each Age")



sns.barplot(x=df_bad_age_dropped.age.sort_values(), y=df.age.value_counts())