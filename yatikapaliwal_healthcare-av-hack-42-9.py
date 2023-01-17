import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data.csv')
test = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/test_data.csv')
train.head()
train.info()
features_with_na=[features for features in train.columns if train[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(train[feature].isnull().mean(), 4),  ' % missing values')
test.isnull().sum().sort_values(ascending= False)
train.shape
test.shape
for i in train.columns:
  print(i,' : ' , train[i].nunique())
for i in test.columns:
  print(i,' : ' , test[i].nunique())
train['Bed Grade'].fillna(train['Bed Grade'].mode()[0], inplace = True)
test['Bed Grade'].fillna(test['Bed Grade'].mode()[0], inplace = True)
train['City_Code_Patient'].fillna(train['City_Code_Patient'].mode()[0], inplace = True)
test['City_Code_Patient'].fillna(test['City_Code_Patient'].mode()[0], inplace = True)
test['Stay'] = -1
df = pd.concat([train, test])
df.shape
from sklearn.preprocessing import LabelEncoder

for i in ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].astype('str'))
train = df[df['Stay']!=-1]
test = df[df['Stay']==-1]
le = LabelEncoder()
train['Stay'] = le.fit_transform(train['Stay'].astype('str'))
def get_countid_enocde(train, test, cols, name):
  temp = train.groupby(cols)['case_id'].count().reset_index().rename(columns = {'case_id': name})
  temp2 = test.groupby(cols)['case_id'].count().reset_index().rename(columns = {'case_id': name})
  train = pd.merge(train, temp, how='left', on= cols)
  test = pd.merge(test,temp2, how='left', on= cols)
  train[name] = train[name].astype('float')
  test[name] = test[name].astype('float')
  train[name].fillna(np.median(temp[name]), inplace = True)
  test[name].fillna(np.median(temp2[name]), inplace = True)
  return train, test
train, test = get_countid_enocde(train, test, ['patientid'], name = 'count_id_patient')
train, test = get_countid_enocde(train, test, ['patientid', 'Hospital_region_code'], name = 'count_id_patient_hospitalCode')
train, test = get_countid_enocde(train, test, ['patientid', 'Ward_Facility_Code'], name = 'count_id_patient_wardfacilityCode')
train.head()
test = test.drop(['Stay', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code'], axis =1)
train = train.drop(['case_id', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code'], axis =1)
import xgboost
classifier=xgboost.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=800, objective='multi:softmax',reg_alpha=0.5, reg_lambda=1.5,
                                 booster='gbtree', n_jobs=4, min_child_weight=2, base_score= 0.75)
X = train.drop('Stay', axis =1)
y = train['Stay']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.25, random_state =100)
classifier.fit(X_train, y_train)
prediction=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(prediction,y_test)
acc_score=accuracy_score(prediction,y_test)
print(acc_score*100)
pred = classifier.predict(test.iloc[:,1:])
submissions = pd.DataFrame(pred, columns=['Stay'])
submissions['case_id'] = test['case_id']
submissions = submissions[['case_id', 'Stay']]
submissions['Stay'] = submissions['Stay'].replace({0:'0-10', 1: '11-20', 2: '21-30', 3:'31-40', 4: '41-50', 5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100', 10: 'More than 100 Days'})
submissions.head()
#from google.colab import files
#submissions.to_csv('submissions_new2.csv', index=False) 
#files.download('submissions_new2.csv')