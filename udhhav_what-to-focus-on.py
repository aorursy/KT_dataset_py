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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler



sns.set_style(style='darkgrid')

pd.set_option('display.max_columns',None)
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.head()
df.shape
df.drop('Serial No.',axis=1, inplace=True)
df.describe()
sns.countplot(df['Research'])
plt.figure(figsize = (10,10))

sns.heatmap(df.corr(),annot=True, cmap='Blues')
df.columns
fts = ['GRE Score', 'TOEFL Score','SOP', 'LOR ', 'CGPA']

for ft in fts:

    sns.distplot(df[ft])

    plt.show()
sns.regplot(x='GRE Score',y='CGPA',data=df)
exams =['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA']

for marks in exams:

    sns.regplot(x=marks,y = 'Chance of Admit ',data=df)

    plt.show()
px.scatter(df, x ='GRE Score',y='Chance of Admit ',color='Research')
x = df.drop('Chance of Admit ',axis=1)

y = df['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state = 10)
print(X_test.shape)

print(X_train.shape)

print(y_test.shape)

print(y_train.shape)
lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)
scores = cross_val_score(lin_reg,X_test,y_test,cv=5,scoring='r2')
scores.mean()
model = LinearRegression(normalize=True)

model.fit(X_test, y_test)

model.score(X_test, y_test)
print('Your chances are {}%'.format(round(model.predict([[305, 108, 4, 4.5, 4.5, 8.35, 0]])[0]*100, 1)))