import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report



%matplotlib inline
df = pd.read_csv('../input/KaggleV2-May-2016.csv')
df.head()
df = pd.get_dummies(df, columns=['Gender','No-show'], drop_first=True)
df['waitDays'] = (pd.to_datetime(df['AppointmentDay']) -

                            pd.to_datetime(df['ScheduledDay'])).dt.days



# The day of the week with Monday=0, Sunday=6

df['Weekday'] = (pd.to_datetime(df['AppointmentDay'])).dt.weekday

df['waitDays'] = df['waitDays'].replace(-1,0)
df = pd.get_dummies(df, columns=['Weekday'], drop_first=True)

df.drop(['ScheduledDay','AppointmentDay','Neighbourhood'],axis=1, inplace=True)

#,'PatientId','AppointmentID'



df.head()
df.describe()
plt.figure(figsize=(12,12))

sns.heatmap(df.corr())
sns.countplot(x='SMS_received', data=df, hue='No-show_Yes')
plt.figure(figsize=(12,8))

sns.boxplot(y='waitDays',x='No-show_Yes', data=df)
X = df.drop('No-show_Yes', axis=1)

y = df['No-show_Yes']
### if oversampling is needed



from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=101)

X_res,y_res = sm.fit_sample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=24)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))