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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import sklearn

import seaborn as sns 
dataset= pd.read_csv('../input/comcast1/Comcast Telecom Complaints data 1.csv')

X = dataset.iloc[:, :-1].values

print(X)



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cust_comp = le.fit_transform(dataset.iloc[:,1])

print(cust_comp)

rec_via = le.fit_transform(dataset.iloc[:,4])

print(rec_via)

city = le.fit_transform(dataset.iloc[:,5])

print(city)

state = le.fit_transform(dataset.iloc[:,6])

print(state)

onbehalf = le.fit_transform(dataset.iloc[:,8])

print(onbehalf)
dataset['year'] = pd.DatetimeIndex(dataset['Date']).year

dataset['month'] = pd.DatetimeIndex(dataset['Date']).month

dataset['day'] = pd.DatetimeIndex(dataset['Date']).day

dataset.drop(["Ticket #", "Time","Zip code","Date"], axis = 1, inplace = True) 



dataset.head()

noc_pm = dataset['month'].value_counts()

print(noc_pm)
noc_pd = dataset['day'].value_counts()

print(noc_pd)
sns.lineplot( data = noc_pm)

plt.title('MONTHLY GRANULARITY LEVELS')

plt.xlabel('DAY')

plt.ylabel('NO. OF COMPLAINTS')

plt.show()
sns.lineplot( data = noc_pd)

plt.title('DAILY GRANULARITY LEVELS')

plt.xlabel('DAY')

plt.ylabel('NO. OF COMPLAINTS')

plt.show()

dataset['Customer Complaint'] = dataset['Customer Complaint'].str.title()

comp = dataset['Customer Complaint'].value_counts()

print(comp)
dataset['Customer Complaint'].value_counts()[:20].plot(kind='bar',figsize=(8,8),stacked=True)
print('The most lodged complaint is : ',dataset.iloc[:,0].value_counts().idxmax())

print('The city that has lodged most number of complaints is :',dataset.iloc[:,2].value_counts().idxmax())

print('The state that has lodged most number of complaints is :',dataset.iloc[:,3].value_counts().idxmax())
dataset["finalstatus"] = ["Open" if Status=="Open" or Status=="Pending" else "Closed"  for Status in dataset["Status"]]

dataset["finalstatus"].unique()

dataset.head()
dataset['State'] = dataset['State'].str.title() 

statecomp = dataset.groupby(['State','finalstatus']).size().unstack().fillna(0)

statecomp
a = statecomp.sort_values("Closed",axis = 0,ascending=False)

a.plot(kind="bar", figsize=(16,12), stacked=True)
print(a)

print('The maximum no of unresolved complaints is:',a.max())
info = dataset.groupby(['Received Via','finalstatus']).size().unstack().fillna(0)

print(info)
total_comp =info['Closed']+info['Open']

print('The total complaints',total_comp)

percentage = info['Closed']/total_comp*100

print('The percentage of complaints resolved',percentage)
year = dataset.iloc[:,6]

month = dataset.iloc[:,7]

day = dataset.iloc[:,8]

import pandas as pd  

lst = [cust_comp,rec_via,city,state,onbehalf,year,month,day]

df_t = pd.DataFrame(lst) 

df = df_t.T

print(df) 


y = dataset.iloc[:,9]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)

print(y)

y = y.reshape(len(y),1)

y

from sklearn.model_selection import train_test_split

df_train, df_test, y_train, y_test = train_test_split(df, y, test_size = 0.25, random_state = 0)



print(df_train)



print(df_test)



print(y_test)



print(y_train)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df_train = sc.fit_transform(df_train)

df_test = sc.fit_transform(df_test)



print(df_test)



print(df_train)

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(df_train, y_train)
y_pred = regressor.predict(df_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test,y_pred)

print(cm)

a = accuracy_score(y_test,y_pred)

print('The accuracy of the model is :',a*100)