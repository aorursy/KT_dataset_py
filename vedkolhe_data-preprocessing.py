import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

df= pd.read_csv('../input/ams-prediction/amsPredictionSheet11.csv')

df.head()
scaler = MinMaxScaler()

df_values = df.values

df_scaled = scaler.fit_transform(df_values)

normalized_df = pd.DataFrame(df_scaled)

normalized_df.head()
normalized_df.describe()
from sklearn.preprocessing import  StandardScaler

std_scaler = StandardScaler()

df_values = df.values

df_std= std_scaler.fit_transform(df_values)

standardized_df = pd.DataFrame(df_std)

standardized_df.head()
standardized_df.describe()
normalized_df.describe()
df= pd.read_csv('../input/std-cat/std cat.csv')

df.head()
df.info()
ew={'F':0,'M':1}

df=df.replace({'Gender':new})

df.info()
df.describe()
df.head()
from sklearn.preprocessing import LabelEncoder



lb = LabelEncoder()

df['State'] = lb.fit_transform(df['State'])

df.head()
df = pd.get_dummies(df, columns=['Category'], prefix = ['Cat'])

df.head()
df.info()
df.describe()
from sklearn.preprocessing import LabelEncoder



lb = LabelEncoder()

df['State'] = lb.fit_transform(df['State'])
df.info()
df.head()
df.describe()