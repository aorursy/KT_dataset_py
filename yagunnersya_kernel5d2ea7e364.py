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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')

df.head()
df.Date = pd.to_datetime(df.Date)

df.set_index('Date',inplace=True)
df.groupby('RainTomorrow').mean()
df.Evaporation.fillna(value=df.Evaporation.mean(),inplace=True)
sns.boxplot(x=df.WindGustDir,y=df.WindGustSpeed) # implying direction isn't as important, hence dropped
df.drop(columns=['WindGustDir','WindDir9am','WindDir3pm','RISK_MM'],inplace=True)
sns.boxplot(x=df.RainTomorrow,y=df.Sunshine)
df.Sunshine.fillna(value='Missing', inplace=True)
def sunshine_fillna(row):

    if row[0] == 'Missing':

        if row[1] == 'Yes':

            return 4.47

        else:

            return 8.55

    else:

        return row[0]
df['Sunshine'] = df[['Sunshine','RainTomorrow']].apply(sunshine_fillna,axis=1)
sns.boxplot(x=df.RainTomorrow,y=df.WindGustSpeed) # similarly for Wind  Gust Speed
df.WindGustSpeed.fillna(value='Missing', inplace=True)
def wgs_fillna(row):

    if row[0] == 'Missing':

        if row[1] == 'Yes':

            return 45.95

        else:

            return 38.29

    else:

        return row[0]
df['WindGustSpeed'] = df[['WindGustSpeed','RainTomorrow']].apply(wgs_fillna,axis=1)
df.Humidity9am.fillna('Missing',inplace=True)

df.Humidity3pm.fillna('Missing',inplace=True)
def hum9am_fillna(row):

    if row[0] == 'Missing':

        if row[1] == 'Yes':

            return 77.98

        else:

            return 66.22

    else:

        return row[0]
def hum3pm_fillna(row):

    if row[0] == 'Missing':

        if row[1] == 'Yes':

            return 68.80

        else:

            return 46.51

    else:

        return row[0]
df['Humidity9am'] = df[['Humidity9am','RainTomorrow']].apply(hum9am_fillna,axis=1)
df['Humidity3pm'] = df[['Humidity3pm','RainTomorrow']].apply(hum3pm_fillna,axis=1)
sns.boxplot(x=df.RainTomorrow,y=df.Pressure9am) # doesn't seem significant, hence dropped
df.drop(columns=['Pressure9am','Pressure3pm'],inplace=True)
df.Cloud9am.fillna('Missing',inplace=True)

df.Cloud3pm.fillna('Missing',inplace=True)
def cloud9am_fillna(row):

    if row[0] == 'Missing':

        if row[1] == 'Yes':

            return 6.1

        else:

            return 3.93

    else:

        return row[0]
def cloud3pm_fillna(row):

    if row[0] == 'Missing':

        if row[1] == 'Yes':

            return 6.36

        else:

            return 3.92

    else:

        return row[0]
df['Cloud9am'] = df[['Cloud9am','RainTomorrow']].apply(cloud9am_fillna,axis=1)
df['Cloud3pm'] = df[['Cloud3pm','RainTomorrow']].apply(cloud3pm_fillna,axis=1)
df.Temp9am.fillna(df.Temp9am.mean(),inplace=True)

df.Temp3pm.fillna(df.Temp3pm.mean(),inplace=True)
sns.boxplot(x=df.RainTomorrow,y=df.MinTemp) # indistinguishable, hence dropped.
df.drop(columns=['MinTemp','MaxTemp'],inplace=True)
sns.boxplot(x=df.RainToday,y=df.WindSpeed9am) # indistinguishable, hence drop both
df.drop(columns=['WindSpeed9am','WindSpeed3pm'],inplace=True)
df.drop(columns=['RainToday'],inplace=True)
df.dropna(inplace=True)
df.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled = scaler.fit_transform(df.drop(columns=['Location','RainTomorrow']))

X = pd.DataFrame(scaled,columns=[['Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',

       'Humidity9am', 'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',

       'Temp3pm']])
X.head()
y = df.RainTomorrow
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
model.score(X_test,y_test)