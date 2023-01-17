# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import findspark
import os
print(os.listdir("../input"))
#import spark
#findspark.init()

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
import datetime

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv("../input/train.csv", nrows=10000000)
df.head()

import random
filename = "../input/train.csv"
n = 1000000 #number of records in file (excludes header)
s = 100000 #desired sample size
skip = random.sample(range(1,n+1),n-s) #the 0-indexed header will not be included in the skip list
dftest = pd.read_csv(filename, skiprows=skip, nrows = 1000000)
# Better sampling techniques are certainly required. This is a just for my trial and learning...

#dftest=pd.read_csv("../input/test.csv", nrows=10000000)
dftest.head()

#make wider graphs
sns.set(rc={'figure.figsize':(20,5)});
plt.figure(figsize=(20,5));
sns.countplot(x='is_attributed', data=df);

# We see that we have very few conversions of the app.
sns.countplot(x='os', data=df);
sns.countplot(x="device", data=df) ; 
sns.countplot(x="app", data=df) ; 
df[["app","is_attributed"]].groupby(["app"]).count().plot() # Frequency of apps  
df[["app","is_attributed"]].groupby(["is_attributed"]).count() # among 10 million clicks, we have 18,717 downloads.
df[["os","is_attributed"]].groupby(["os"], as_index=False).count().plot()
# Most used os to least used OS :

df[["app","is_attributed"]].groupby(["app"], as_index=False).count().sort_values("is_attributed", ascending = False)
df[["app","is_attributed"]].groupby(["app"], as_index=False).mean().sort_values("is_attributed", ascending = False)
df[["device","is_attributed"]].groupby(["device"], as_index=False).count().sort_values("is_attributed", ascending = False)
df[["device","is_attributed"]].groupby(["device"], as_index=False).count().plot()
df['click_time'] = pd.to_datetime(df['click_time'])
df['attributed_time'] = pd.to_datetime(df['attributed_time'])
dftest['click_time'] = pd.to_datetime(dftest['click_time'])
df['hr']=df['click_time'].dt.hour
dftest['hr']=dftest['click_time'].dt.hour
#dftest['attributed_time'] = pd.to_datetime(dftest['attributed_time'])
#datetime.hour
# dt.round('H')   


df[['hr','is_attributed']].groupby(['hr'], as_index=True).count().plot()
plt.title('HOURLY CLICK FREQUENCY');
plt.ylabel('Number of Clicks');



dftest.head()
x = df.drop(['is_attributed','click_time','ip','attributed_time'], axis=1)
y = df['is_attributed']
dftestx = dftest.drop(['click_time','ip','attributed_time','is_attributed'], axis = 1)
ytest = dftest['is_attributed']
x.head();
dftestx.head()
scaler = MinMaxScaler()
scaler.fit(x)
xscaled = scaler.transform(x)
scaler.fit(dftestx)

dftestxscaled = scaler.transform(dftestx)
# The have become numpy arrays
#x.head()
#dftestxscaled.head()
pd.DataFrame(xscaled)
pd.DataFrame(dftestxscaled);
#xscaled.head()
nn = MLPClassifier(hidden_layer_sizes=(5,5,5,5,5), activation='logistic', max_iter=10000000, solver='lbfgs')
nn.fit(x, y)
print(accuracy_score (ytest, nn.predict(dftestx)))
print(" Thanks for listening! :)")
