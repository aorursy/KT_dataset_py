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
train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
id = test['id']
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train.head()
test.head()
train.head()
train.describe()
train.isnull().sum()
train['Response'].value_counts()
# Around 12% of the customers agreed to have a vehicle insurance
percentage = (train[train['Response']==1].count())/(train['Response'].count())*100
print(percentage['Response'])
sns.countplot(train['Response'])
# As we can see most of the customers are between age 20 and 30
sns.distplot(train['Age'])
twenties = train[(train['Age'] > 20) & (train['Age'] < 30)]['Age']
twenties.value_counts().sum()
train.shape
# Until age of 30 customers were not very interested with the vehicle insurance
plt.figure(figsize = (25,10))
sns.countplot(data = train,x = 'Age',hue = 'Response')
# If a customer has already a vehicle insurance 95% percent of them did not interested to have vehicle insurance
plt.figure(figsize = (25,10))
sns.countplot(data = train,x = 'Previously_Insured',hue = 'Response')
train.head(1)
plt.figure(figsize=(10,5))
sns.countplot(data=train,x='Gender',hue='Response')
plt.figure(figsize = (15,7))
sns.scatterplot(y = 'Age',x = 'Annual_Premium',data = train)
plt.figure(figsize = (5,7))
sns.boxplot(data = train, y = 'Annual_Premium')
train.head(1)
train['Vehicle_Age'].value_counts()
plt.figure(figsize = (10,5) )
sns.countplot(x = 'Vehicle_Age',hue = 'Response',data = train)
train.groupby(['Vehicle_Age','Response'])['Response'].count()
train.head()
df=train.groupby(['Vehicle_Damage','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
df
sns.catplot(x = 'Vehicle_Damage',col = 'Response',y = 'count',kind = 'bar',data = df)
train['test'] = 0
test['test'] = 1
df = pd.concat([train,test],axis = 0)

#we can drop id and region code as they will have no effect on response
df.drop(['Region_Code','Policy_Sales_Channel'],axis=1,inplace=True)
df.isnull().sum()
df.loc[df['Gender'] == 'Male', 'Gender'] = 1
df.loc[df['Gender'] == 'Female', 'Gender'] = 0

df.loc[df['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0
df.loc[df['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1
df.loc[df['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2

df.loc[df['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1
df.loc[df['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0
df
correlations = df.corr()
f, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(correlations, annot = True)
test =df[df['test'] == 1]
train = df[df['test'] == 0]
test.drop(['test'],axis =1 , inplace = True)
train.drop(['test'],axis =1 , inplace = True)
train.drop(['id'],axis =1 , inplace = True)
test.drop(['id'],axis =1 , inplace = True)
test
train
y = train['Response']
train.drop(['Response'],axis = 1 ,inplace = True)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(train , y , test_size = 0.3)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
model1 = LogisticRegression()
model1.fit(x_train,y_train)

y_pred1 = model1.predict(x_test)
print("Accuracy {} %".format( 100 * accuracy_score(y_pred1, y_test)))
cm = confusion_matrix(y_pred1, y_test)
sns.heatmap(cm, annot=True)
cm
print(classification_report(y_test, y_pred1))
# Accuracy is 87 and std is low so this is a model we can trust
from sklearn.model_selection import cross_val_score
cross_val1 = cross_val_score(estimator = model1 , X = x_train , y = y_train , cv = 10)
print("Accuracy: {:.2f} %".format(cross_val1.mean()*100))
print("Standard Deviation: {:.2f} %".format(cross_val1.std()*100))
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
print("Accuracy {} %".format( 100 * accuracy_score(y_pred2, y_test)))
cross_val2 = cross_val_score(estimator = model2 , X = x_train , y = y_train , cv = 10)
print("Accuracy: {:.2f} %".format(cross_val2.mean()*100))
print("Standard Deviation: {:.2f} %".format(cross_val2.std()*100))
test.isnull().sum()
test.drop(['Response'],axis =1 ,inplace=True)

pred = model1.predict(test)
submission = pd.DataFrame(data = {'id': id, 'Response': pred})
submission.to_csv('vehicle_insurance_catboost.csv', index = False)
submission.head()
