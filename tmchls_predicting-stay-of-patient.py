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
#loading the datasets
train=pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv')
test=pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/test_data.csv')
train.head()
test.head()
#renaming some of the features in train
train.rename(columns={'Available Extra Rooms in Hospital':'Extra_Rooms','Type of Admission':'Admission_Type',
                     'Severity of Illness':'Illness_severity','Visitors with Patient':'Visitors'},inplace=True)
#renaming some of the features in test
test.rename(columns={'Available Extra Rooms in Hospital':'Extra_Rooms','Type of Admission':'Admission_Type',
                     'Severity of Illness':'Illness_severity','Visitors with Patient':'Visitors'},inplace=True)
train.info()
train.describe(include='all')
#converting object type columns into categorical type
columns=['Hospital_type_code','Hospital_region_code','Department', 'Ward_Type',
       'Ward_Facility_Code','Admission_Type', 'Illness_severity', 'Age',
       'Stay']
for i in columns:
    train[i]=train[i].astype('category')
    if i!='Stay':
        test[i]=train[i].astype('category')
train.info()
#checking for missing values in test and train dataset
train.isnull().sum()
test.isnull().sum()
#filling the missing value in bed grade column in both train and test
train['Bed Grade']=train['Bed Grade'].fillna(1.0)
test['Bed Grade']=test['Bed Grade'].fillna(1.0)
train.drop(columns=['City_Code_Patient'],inplace=True)
test.drop(columns=['City_Code_Patient'],inplace=True)
train.isnull().sum()
test.isnull().sum()
#importing libraries for visualization
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
plotly.offline.init_notebook_mode(connected = True)
train.groupby('Department')['Extra_Rooms'].agg('count')
train.groupby('Bed Grade')['Extra_Rooms'].agg('count')
train.groupby('Admission_Type')['Extra_Rooms'].agg('count')
train.groupby('Illness_severity')['Extra_Rooms'].agg('count')
train.groupby('Department')['Bed Grade'].agg('mean')
train.groupby('Admission_Type')['Bed Grade'].agg('mean')
train.groupby('Illness_severity')['Bed Grade'].agg('mean')
px.pie(train,values='Extra_Rooms',names='Department',title='Distribution of Extra Rooms in Departments')
px.pie(train,values='Extra_Rooms',names='Bed Grade',title='Distribution of Bed in extra rooms')
px.pie(train,values='patientid',names='Age',title='Distribution of Age in Patients')
px.pie(train,values='patientid',names='Stay',title='Distribution of Stay Length of Patients')
train.columns
#taking categorical variables to label encode
cat_columns=['Age','Stay']
#storing the encoded values both in train and test sets
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in cat_columns:
    train[i]=l.fit_transform(train[i])
    if i!='Stay':
        test[i]=l.transform(test[i])
#generating one hot features of remaining categorical features
train=pd.get_dummies(train)
test=pd.get_dummies(test)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
#dropping irrelevant columns 
X=train.drop(columns=['Stay'])
Y=train['Stay']
X.head()
#dividing data into train and test sets
X_train,X_valid,y_train,y_valid=train_test_split(X,Y,test_size=0.2,random_state=0)
#scaling the features
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_valid=sc.transform(X_valid)
from catboost import Pool, CatBoostClassifier
eval_dataset = Pool(data=X_valid, label=y_valid)
# initialising catboost classifier

model = CatBoostClassifier(iterations=500,learning_rate=0.3,
                           eval_metric='Accuracy')
model.fit(X_train,y_train,eval_set=eval_dataset)
model.get_best_score()
test_dataset=Pool(test)
y_pred=model.predict(test_dataset)
y_pred