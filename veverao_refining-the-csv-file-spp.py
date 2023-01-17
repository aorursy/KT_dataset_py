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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('whitegrid')

from sklearn.preprocessing import normalize,StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error,accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

df= pd.read_csv("../input/new123/refined1.csv")
df.drop(["reason","higher","famrel"],axis=1,inplace=True)

df


new_data = df.rename(columns = {"social": "socialising","address":"addresstype","sex":"gender","nursery":"elementaryschool","Final":"result1sem"}) 

new_data.head()


new_data['Tenthg'] = 'na'

new_data.loc[(new_data.Tenth >= 16) & (new_data.Tenth <= 20), 'Tenthg'] = 'excellent' 

new_data.loc[(new_data.Tenth >= 11) & (new_data.Tenth <= 15), 'Tenthg'] = 'good' 

new_data.loc[(new_data.Tenth >= 6) & (new_data.Tenth <= 10), 'Tenthg'] = 'average' 

new_data.loc[(new_data.Tenth >= 0) & (new_data.Tenth <= 5), 'Tenthg'] = 'poor' 





new_data['Twelfthg'] = 'na'

new_data.loc[(new_data.Twelfth >= 16) & (new_data.Twelfth <= 20), 'Twelfthg'] = 'excellent' 

new_data.loc[(new_data.Twelfth >= 11) & (new_data.Twelfth <= 15), 'Twelfthg'] = 'good' 

new_data.loc[(new_data.Twelfth >= 6) & (new_data.Twelfth<= 10), 'Twelfthg'] = 'average' 

new_data.loc[(new_data.Twelfth >= 0) & (new_data.Twelfth <= 5), 'Twelfthg'] = 'poor' 





new_data['grade1sem'] = 'na'

new_data.loc[(new_data.result1sem >= 16) & (new_data.result1sem <= 20), 'grade1sem'] = 'excellent' 

new_data.loc[(new_data.result1sem >= 11) & (new_data.result1sem <= 15), 'grade1sem'] = 'good' 

new_data.loc[(new_data.result1sem >= 6) & (new_data.result1sem <= 10), 'grade1sem'] = 'average' 

new_data.loc[(new_data.result1sem >= 0) & (new_data.result1sem <= 5), 'grade1sem'] = 'poor' 



new_data.drop(["Tenth","Twelfth","result1sem"],axis=1,inplace=True)

new_data



new_data.tail(5)
new_data.to_csv('dataset.csv', index=False) 
df1=pd.read_csv("../input/workingdata/dataset.csv")

df1.describe()

df1["failures"].head(10)
df1.isnull().sum()

df1.skew(axis=0)
df1["traveltime"] = np.log(df1["traveltime"])

df1["traveltime"].skew()
df1["studytime"] = np.sqrt(df1["studytime"])

df1["studytime"].skew()
df1["failures"] = np.sqrt(df1["failures"])

df1["failures"].skew()





df1["absences"] = np.sqrt(df1["absences"])

df1["absences"].skew()
df1.skew(axis=0)
sns.heatmap(df1.corr())

plt.show()
# Final Grade Countplot

plt.figure(figsize=(8,6))

sns.countplot(df1.grade1sem, order=["poor","average","good","excellent"], palette='Set1')

plt.title('1 semester Grade - Number of Students',fontsize=20)

plt.xlabel('1 semester Grade', fontsize=16)

plt.ylabel('Number of Students', fontsize=16)
df1.info()
df1=pd.get_dummies(df1, columns = ["gender","addresstype","famsize","Pstatus","Mjob","Fjob","guardian","schoolsup","famsup","tution","activities","elementaryschool","internet","Tenthg","Twelfthg"],drop_first=True)

df1.head()
scaler = StandardScaler()

scaler.fit(df1)

scaler.transform(df1)

nm = normalize(df1)

#cols = df1.columns

#df1 = pd.DataFrame(nm, columns = cols)

#x = df1.drop(["grade1sem_excellent","grade1sem_good","grade1sem_poor"], axis = 1)

#y = df1[["grade1sem_excellent","grade1sem_good","grade1sem_poor"]]
cols = df1.columns

df1 = pd.DataFrame(nm, columns = cols)

x = df1.drop(["grade1sem"], axis = 1)

y = df1[["grade1sem"]]
#correlation matrix

corrmat = df1.corr()

f, ax = plt.subplots(figsize=(150, 50))

sns.set(font_scale=1.25)

sns.heatmap(corrmat, annot=True, vmax=1.0, square=True)

plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state =10)
x_train
y_train
regr = RandomForestClassifier()

regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

print("Accuracy:",accuracy_score(y_test, y_pred))