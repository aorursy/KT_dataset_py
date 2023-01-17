import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# importing some libraries that will help me plot the data

import seaborn as sns

import matplotlib.pyplot as plt



# importing libraries to preprocess the data

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")

data.head()
sns.set_palette([ 'pink','navy'])

sns.set_context("poster", font_scale = 0.8)

plt.figure(figsize=(9,9))

sns.countplot(data['gender'])

plt.xlabel("Gender")

plt.ylabel("Count")
sns.set_palette("GnBu_d")

sns.set_context("poster", font_scale = 0.6)

plt.figure(figsize=(15,7))

sns.countplot(data['parental level of education']) 

plt.xlabel('Parental Level of Education')

plt.ylabel('Count')
sns.set_context("poster", font_scale = 0.8)

plt.figure(figsize=(9,9))

sns.set_palette([ 'orange','red'])

sns.scatterplot(data=data, x='reading score', y='writing score', hue='test preparation course' )
sns.set_context("poster", font_scale = 0.8)

plt.figure(figsize=(9,9))

sns.set_palette([ 'pink','navy'])

sns.scatterplot(data=data, x='math score', y='writing score', hue='gender' )
c = (data.dtypes == 'object')

cat_col = list(c[c].index)

print(cat_col)
enc = LabelEncoder()



for col in cat_col:

    data[col] = enc.fit_transform(data[col])



data.head()
plt.figure(figsize=(10,10))

sns.set_context("poster", font_scale = 0.5)

sns.heatmap(data.corr(), cmap="Blues")
data = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
sns.catplot(data=data, x='race/ethnicity',y='math score', hue='gender', kind="bar")
columns = data.columns

enc = LabelEncoder()

for col in cat_col:

    data[col] = enc.fit_transform(data[col])



encdata = pd.DataFrame(data, columns=columns)    



y = encdata['gender']

encdata.drop(['gender'], inplace=True, axis=1)



xtrain, xtest, ytrain, ytest  = tts(encdata, y, train_size=0.7, test_size=0.3)
model = SVC()

model.fit(xtrain, ytrain)



preds = model.predict(xtest)



print("The Accuracy of this model is:", accuracy_score(preds, ytest)*100,"%")