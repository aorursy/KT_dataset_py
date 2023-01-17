import pandas as pd

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("../input/machine-learning/amsPredictionSheet11-201010-101537.csv") 

df.head() 
df.columns
df.head(24)   #top_rows_in_given_ds
df.tail()   #bottom_side
Scaler=MinMaxScaler()

df_values=df.values

df_valued=Scaler.fit_transform(df_values)

normalized_df=pd.DataFrame(df_valued) 

normalized_df
df.describe()  #to find mean,count,std deviation,min and max value

                  
std_scaler=StandardScaler() 
df_values=df.values

df_std=std_scaler.fit_transform(df_values)

std_df=pd.DataFrame(df_std)

std_df
df.describe() 
print(df.isnull().sum()) 
df.Attendance=df.Attendance.fillna("student")   #null_values

print(df.isnull().sum())
print(df.shape)

print("\n")

print(df.dtypes)
df.info()

print('_'*24)
df=pd.read_csv("../input/ipl-data-set/matches.csv")

df.head() 
df.info() 
lb=LabelEncoder()

df['date'] =lb.fit_transform(df['date'])

df.head() 
df.info()
new={'No':0,'yes':1}

df.id=df.id.map(new)

df.head()
df.info() 