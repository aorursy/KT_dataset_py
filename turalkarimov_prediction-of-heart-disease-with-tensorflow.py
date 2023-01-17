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
df = pd.read_csv(r"/kaggle/input/heart-disease-uci/heart.csv")
df.head()
#Data cleaing
df.isnull().sum()
df.dtypes
#data visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.countplot(x='target', data=df)
sns.countplot(x='target',hue='sex', data=df)

plt.xlabel("Gender (0 = female, 1= male)")
plt.show()
disease_no = df[df.target==1].age
disease_yes = df[df.target==0].age

plt.xlabel("Ages")
plt.ylabel("Number Of Patients")
plt.title("Patients Disease Prediction Visualiztion")

plt.hist([disease_yes, disease_no], rwidth=0.95, color=['red','blue'],label=['Disease=Yes','Disease=No'])
plt.legend()
#preparation for training
col = df.columns
for i in col:
    print(i, df[i].unique()[:10])
#for traing we need get dummies for cp,restecg,slope,ca,thal columns
cp = pd.get_dummies(df["cp"])
restecg = pd.get_dummies(df["restecg"])
slope = pd.get_dummies(df["slope"])
ca = pd.get_dummies(df["ca"])
thal = pd.get_dummies(df["thal"])
df1 = df.drop(['cp','restecg','slope','ca','thal'],axis=1)
df1.head()
#for getting high score during training we will use Minmax sclaer for age,trestbps,chol,thalach (machine can better read values between [0:1])

from sklearn.preprocessing import MinMaxScaler
mx = MinMaxScaler()
df1['age'] = mx.fit_transform(df1[['age']])
df1['age'] = mx.fit_transform(df1[['age']])
df1['trestbps'] = mx.fit_transform(df1[['trestbps']])
df1['chol'] = mx.fit_transform(df1[['chol']])
df1['thalach'] = mx.fit_transform(df1[['thalach']])
df1['oldpeak'] = mx.fit_transform(df1[['oldpeak']])
df2 = pd.concat([df1,cp,restecg,slope,ca,thal],axis=1)
df2.head(2)
#now our model ready for traing 
X = df2.drop('target',axis=1)
y = df2['target']
from sklearn.model_selection import train_test_split as tts

X_train,X_test,y_train,y_test = tts(X,y,test_size=0.2)

import tensorflow as tf
import keras 
model = keras.Sequential([
    keras.layers.Dense(16,input_shape=(X_train.shape[1],),activation="relu"),
    keras.layers.Dense(64,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100)
model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred[:5]
#for reading predited data we will do :
y_predicted=[]
for num in y_pred:
    if num>0.5:
        y_predicted.append(1)
    else:
        y_predicted.append(0)
y_predicted[:7]
from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_predicted))
confusion_matrix = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted)

plt.figure(figsize = (10,7))
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
#Thanks for review!!
