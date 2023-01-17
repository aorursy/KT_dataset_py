import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

train_data = pd.read_csv('../input/janatahack-healthcare-analytics-ii-dataset/train.csv')

test_data = pd.read_csv('../input/janatahack-healthcare-analytics-ii-dataset/test.csv')

combined = [train_data,test_data]

train_data
le = LabelEncoder()

train_data['Stay'] = le.fit_transform(train_data['Stay'])
train_data.columns
train_data.info()
train_data.describe()
for dataset in combined:

    dataset.drop(['case_id','patientid'],axis = 1,inplace = True)
train_data[['Hospital_code','Stay']].groupby('Hospital_code').mean()
train_data[['City_Code_Hospital','Stay']].groupby('City_Code_Hospital').mean()
train_data[['Hospital_type_code','Stay']].groupby('Hospital_type_code').mean()
train_data[['Hospital_region_code','Stay']].groupby('Hospital_region_code').mean()
train_data[['Available Extra Rooms in Hospital','Stay']].groupby('Available Extra Rooms in Hospital').count()
train_data[['Available Extra Rooms in Hospital','Stay']].groupby('Available Extra Rooms in Hospital').mean()
train_data[['Department','Stay']].groupby('Department').mean()
train_data[['Ward_Type','Stay']].groupby('Ward_Type').mean()
train_data[['Ward_Facility_Code','Stay']].groupby('Ward_Facility_Code').mean()
train_data[['Type of Admission','Stay']].groupby('Type of Admission').mean()
train_data[['Severity of Illness','Stay']].groupby('Severity of Illness').mean()

train_data[['Visitors with Patient','Stay']].groupby('Visitors with Patient').mean()

train_data['Visitors with Patient'].nunique()
train_data[['Age','Stay']].groupby('Age').mean()

train_data['Age'] = le.fit_transform(train_data['Age'])

test_data['Age'] = le.transform(test_data['Age'])
train_data['Bed Grade'] = train_data['Bed Grade'].fillna(train_data['Bed Grade'].mean()).astype(int)

test_data['Bed Grade'] = test_data['Bed Grade'].fillna(test_data['Bed Grade'].mean()).astype(int)


train_data[['Admission_Deposit','Stay']].groupby('Admission_Deposit').mean()

train_data['CategoricalAdmission_Deposit'] = pd.qcut(train_data['Admission_Deposit'], 5)

train_data[['CategoricalAdmission_Deposit','Stay']].groupby('CategoricalAdmission_Deposit').mean()

train_data.loc[train_data['Admission_Deposit'] <= 4051,'Admission_Deposit'] = 0

train_data.loc[(train_data['Admission_Deposit'] > 4051) & (train_data['Admission_Deposit'] <= 4528),'Admission_Deposit'] = 1

train_data.loc[(train_data['Admission_Deposit'] > 4528) & (train_data['Admission_Deposit'] <= 4968),'Admission_Deposit'] = 2

train_data.loc[(train_data['Admission_Deposit'] > 4968) & (train_data['Admission_Deposit'] <= 5611),'Admission_Deposit'] = 3

train_data.loc[(train_data['Admission_Deposit'] > 5611) & (train_data['Admission_Deposit'] <= 11008),'Admission_Deposit'] = 4
test_data.loc[test_data['Admission_Deposit'] <= 4051,'Admission_Deposit'] = 0

test_data.loc[(test_data['Admission_Deposit'] > 4051) & (test_data['Admission_Deposit'] <= 4528),'Admission_Deposit'] = 1

test_data.loc[(test_data['Admission_Deposit'] > 4528) & (test_data['Admission_Deposit'] <= 4968),'Admission_Deposit'] = 2

test_data.loc[(test_data['Admission_Deposit'] > 4968) & (test_data['Admission_Deposit'] <= 5611),'Admission_Deposit'] = 3

test_data.loc[(test_data['Admission_Deposit'] > 5611) & (test_data['Admission_Deposit'] <= 11008),'Admission_Deposit'] = 4
train_data.drop('CategoricalAdmission_Deposit',axis = 1,inplace = True)
for dataset in combined:

    dataset['City_Code_Patient'].fillna(8.0,inplace = True)
for dataset in combined:

    dataset['City_Code_Patient'] = dataset['City_Code_Patient'].astype(int)
train_data['Admission_Deposit'] = train_data['Admission_Deposit'].astype(int)

train_data
test_data['Admission_Deposit'] = test_data['Admission_Deposit'].astype(int)

test_data
cat_features = train_data.select_dtypes(['object']).columns

cat_features
train_data = pd.get_dummies(train_data,columns = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type',

       'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness'])
test_data = pd.get_dummies(test_data,columns = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type',

       'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness'])
plt.figure(figsize = (16,12))

sns.heatmap(train_data.corr(),annot=True,cbar = True)

plt.show()
y_train = train_data.loc[:,'Stay']

x_train = train_data.drop('Stay',axis = 1)

x_train
clf = OneVsRestClassifier(XGBClassifier()).fit(x_train, y_train)
clf.score(x_train,y_train)
y_pred = clf.predict(test_data)

y_pred
sub = pd.DataFrame(y_pred,columns = ['Stay'],index = [i + 318439 for i in range(test_data.shape[0])])



sub.index.name = 'case_id'
map_dict = {

    0: '0-10',

    1: '11-20',

    2: '21-30',

    3: '31-40',

    4: '41-50',

    5: '51-60',

    6: '61-70',

    7: '71-80',

    8: '81-90',

    9: '91-100',

    10: 'More than 100 Days'

}

sub['Stay'] = sub['Stay'].map(map_dict)

sub
sub.to_csv('latest.csv')