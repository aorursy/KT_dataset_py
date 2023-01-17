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
df = pd.read_csv('/kaggle/input/heart-attack-prediction/data.csv')

df.head()
df.isnull().sum()
x = df.drop(columns=['slope','ca','thal','fbs'])

print(x)
y = df['fbs']

print(y)
from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()  

x= x.apply(label_encoder.fit_transform)

print(x)
y= label_encoder.fit_transform(y)

print(y)
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go
iplot([go.Histogram2dContour(x=df.head(500)['age'], 

                             y=df.head(500)['fbs'], 

                             contours=go.Contours(coloring='heatmap')),

       go.Scatter(x=df.head(1000)['age'], y=df.head(1000)['fbs'], mode='markers')])
df = df.assign(n=0).groupby(['fbs', 'age'])['n'].count().reset_index()

df = df[df["age"] < 100]

v = df.pivot(index='age', columns='fbs', values='n').fillna(0).values.tolist()
iplot([go.Surface(z=v)])
import seaborn as sns

import matplotlib.pyplot as plt
# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Heart attack prediction")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=df['fbs'], y=df['age'])



# Add label for vertical axis

plt.ylabel("age")
sns.lineplot(data=y)
sns.scatterplot(data=df, x="age", y="fbs")
df['fbs'].value_counts().head(10).plot.pie()
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
nb = GaussianNB()

nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

print(y_pred)



print(accuracy_score(y_test, y_pred))
from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# random forest model creation

rfc = RandomForestClassifier()

rfc.fit(x_train,y_train)

# predictions

rfc_predict = rfc.predict(x_test)



print("Accuracy:",accuracy_score(y_test, rfc_predict))