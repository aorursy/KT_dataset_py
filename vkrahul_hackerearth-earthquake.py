# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.listdir('../input')
# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/earthquake-data-from-hackerearth/train.csv')
test = pd.read_csv('../input/test-data/test.csv')
structure = pd.read_csv('../input/earthquake-data-from-hackerearth/Building_Structure.csv')
ownership = pd.read_csv('../input/earthquake-data-from-hackerearth/Building_Ownership_Use.csv')
merged = pd.merge(structure, ownership,how='inner',left_on=['building_id','district_id', 'vdcmun_id', 'ward_id'],right_on=['building_id','district_id', 'vdcmun_id', 'ward_id'])
train.shape
test.shape
structure.shape
ownership.shape
merged.shape
train_all = pd.merge(left=train,right=merged,left_on=['building_id','district_id', 'vdcmun_id'],right_on=['building_id','district_id', 'vdcmun_id'])
test_all = pd.merge(left=test,right=merged,left_on=['building_id','district_id', 'vdcmun_id'],right_on=['building_id','district_id', 'vdcmun_id'])
train_all.drop('has_repair_started',axis=1,inplace=True)
test_all.drop('has_repair_started',axis=1,inplace=True)
train_all.shape
test_all.shape
features = list(train_all.columns)
features.remove('damage_grade')
# features.remove('damage_label')
features.remove('building_id')
features.remove('district_id')
features.remove('ward_id')
features.remove('vdcmun_id')

len(features)
train_all.head()
train_all['damage_label'] = train_all.apply(lambda x: np.where(x['damage_grade'] == 'Grade 1',1,np.where(x['damage_grade'] == 'Grade 2',2,np.where(x['damage_grade'] == 'Grade 3',3,np.where(x['damage_grade'] == 'Grade 4',4,np.where(x['damage_grade'] == 'Grade 5',5,0))))),axis=1)

train_all['damage_label'] = train_all['damage_label'].map(lambda x: x)
temp_features = features[:]
temp_features.remove('legal_ownership_status')
temp_features.remove('area_assesed')
temp_features.remove('land_surface_condition')
temp_features.remove('foundation_type')
temp_features.remove('roof_type')
temp_features.remove('ground_floor_type')
temp_features.remove('other_floor_type')
temp_features.remove('position')
temp_features.remove('plan_configuration')
temp_features.remove('condition_post_eq')

to_be_dummied = [x for x in features if x not in temp_features ]
to_be_dummied
train_all.dropna(inplace=True)
train_all.shape

train_all = pd.get_dummies(train_all,columns=to_be_dummied)
test_all = pd.get_dummies(test_all,columns=to_be_dummied)
# test_all.drop(['building_id','district_id','ward_id','vdcmun_id'],axis=1,inplace=True)

features = list(train_all.columns)
features.remove('damage_label')
features.remove('damage_grade')
features.remove('building_id')
features.remove('district_id')
features.remove('ward_id')
features.remove('vdcmun_id')
features
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X = train_all[features]
y=train_all['damage_label']
# X_train,X_test,y_train,y_test = train_test_split(train_all[features],train_all['damage_label'],random_state=0)
# X_train,X_test,y_train,y_test =0,0,0,0
# from sklearn.model_selection import KFold 
# kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None) 

# for train_index, test_index in kf.split(X):
#     print("Train:", train_index, "Validation:",test_index)
#     X_train, X_test = X[train_index], X[test_index] 
#     y_train, y_test = y[train_index], y[test_index]
# from sklearn.svm import SVC
# clf = SVC().fit(X,y)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X,y)
from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier().fit(X_train,y_train)
rf = RandomForestClassifier().fit(X,y)
X_test = test_all[features]
rf_predicted = rf.predict(X_test)
svc_predicted = clf.predict(X_test)
# print(classification_report(y_test,rf_predicted))
rf_predicted
final_values_rf=[]
for i in rf_predicted:
    if(i==1):
        final_values_rf.append('Grade 1')
    elif(i==2):
        final_values_rf.append('Grade 2')
    elif(i==3):
        final_values_rf.append('Grade 3')
    elif(i==4):
        final_values_rf.append('Grade 4')
    elif(i==5):
        final_values_rf.append('Grade 5')
final_values_svc=[]
for i in svc_predicted:
    if(i==1):
        final_values_svc.append('Grade 1')
    elif(i==2):
        final_values_svc.append('Grade 2')
    elif(i==3):
        final_values_svc.append('Grade 3')
    elif(i==4):
        final_values_svc.append('Grade 4')
    elif(i==5):
        final_values_svc.append('Grade 5')
submission = pd.DataFrame({'building_id':test_all['building_id'],'damage_grade':final_values_rf})
submission.head()
submission2 = pd.DataFrame({'building_id':test_all['building_id'],'damage_grade':final_values_svc})
submission2.head()
filename='EarthQuake predictions 1.csv'
submission.to_csv(filename,index=False)
print("saved file:"+filename)
filename2='EarthQuake predictions 2.csv'
submission2.to_csv(filename2,index=False)
print("saved file:"+filename2)