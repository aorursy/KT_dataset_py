
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# These two bad boys will help us plot our data and understand it a bit graphically:
import seaborn as sns
import matplotlib.pyplot as plt

# Importing some tools to preprocess the data:
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Lets import some algorithms now:
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score

# Importing tools form Keras library in order to build Neural Network:
from keras.losses import binary_crossentropy
from keras.layers import Dense
from keras.models import Sequential
from keras.metrics import Accuracy
from keras.optimizers import RMSprop, SGD, Adam
from keras.optimizers.schedules import ExponentialDecay 

data = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv", index_col='customerID')
data.head()

data.isnull().sum()
plt.figure(figsize=(7,7))
sns.set_context("poster", font_scale=0.7)
sns.set_palette(['pink', 'skyblue'])
sns.countplot(data['gender'])

plt.figure(figsize=(8,8))
sns.set_context("poster", font_scale=0.7)
sns.set_palette(['k', 'darkgrey'])
sns.countplot(data['SeniorCitizen'])
plt.xticks([0,1], ['Not a SeniorCitizen', 'SeniorCitizen'])
c = (data.dtypes == 'object')
catcol = list(c[c].index)
encdata = data.copy()
enc = LabelEncoder()
columns = data.columns
for col in catcol:
    encdata[col] = enc.fit_transform(encdata[col])
    
encdata = pd.DataFrame(encdata, columns=columns)
plt.figure(figsize=(9,9))
sns.heatmap(encdata.corr(), cmap='Blues')
plt.figure(figsize=(7,7))
sns.set_context("poster", font_scale=0.7)
sns.set_palette(['pink', 'skyblue'])
sns.scatterplot(data=data, x='TotalCharges', y='tenure', hue='Churn')
plt.figure(figsize=(7,7))
sns.set_context("poster", font_scale=0.7)
sns.set_palette(['pink', 'skyblue'])
sns.scatterplot(data=data, x='MonthlyCharges', y='tenure', hue='Churn')
data = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()
y = data['Churn']

enc = LabelEncoder()
y = enc.fit_transform(y)

data.drop(['Churn', 'customerID'], axis=1, inplace=True)
c = (data.dtypes == 'object')
catcol = list(c[c].index)
for col in catcol:
    data[col] = enc.fit_transform(data[col])
xtrain, xtest, ytrain, ytest = train_test_split(data, y, train_size=0.95, test_size=0.05)
DecModel = DecisionTreeClassifier()

DecModel.fit(xtrain, ytrain)

DecPreds = DecModel.predict(xtest)

accuracy_score(DecPreds, ytest)
DecModel = RandomForestClassifier(n_estimators=1500)

DecModel.fit(xtrain, ytrain)

DecPreds = DecModel.predict(xtest)

accuracy_score(DecPreds, ytest)
def neuralnet(xtrain, xtest, ytrain, ytest):
    NModel = Sequential([
    Dense(128, input_shape=(19,), activation='relu'),
    Dense(240, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
    ])
    
    adam = Adam(learning_rate=0.007)
    
    NModel.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    Fit = NModel.fit(xtrain, ytrain, epochs=50, validation_data=(xtest, ytest))
    return Fit
scale = StandardScaler()

scaledtrain = scale.fit_transform(xtrain)
scaledtest = scale.transform(xtest)
neuralnet(scaledtrain, scaledtest, ytrain, ytest)