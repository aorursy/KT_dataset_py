# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
###################################################
pd.set_option('display.expand_frame_repr',False)
Train = pd.read_csv("../input/saftey_efficay_myopiaTrain.csv")
Test = pd.read_csv("../input/saftey_efficay_myopiaTest.csv")
print(Train.shape,Test.shape)
X_train = Train.iloc[:30451, :-1]
y_train = Train.iloc[:30451, -1:]
X_test = Test.iloc[:, :]

X_train.rename(columns={'T_L_Therapeutic_Cont._L.':'T_L_Therapeutic_Cont_L'},inplace=True)
X_train.rename(columns={'T_L_Cust._Ablation':'T_L_Cust_Ablation'},inplace=True)
X_train.rename(columns={'T_L_Max_Abl._Depth':'T_L_Max_Abl_Depth'},inplace=True)
X_train.rename(columns={'T_L_Op.Time':'T_L_Op_Time'},inplace=True)

X_test.rename(columns={'T_L_Therapeutic_Cont._L.':'T_L_Therapeutic_Cont_L'},inplace=True)
X_test.rename(columns={'T_L_Cust._Ablation':'T_L_Cust_Ablation'},inplace=True)
X_test.rename(columns={'T_L_Max_Abl._Depth':'T_L_Max_Abl_Depth'},inplace=True)
X_test.rename(columns={'T_L_Op.Time':'T_L_Op_Time'},inplace=True)
count = 0


####################################################



#fill all the NaN with median
def fillMedian(att):
    att = att.replace(np.NaN,att.median())
    att = (att-att.mean())/att.std()
    return att


#fill all the NaN with mean
def fillMean(att):
    att = att.replace(np.NaN,att.mean())
    att = (att - att.mean()) / att.std()
    return att


#fill all NaN with 1/mean
def fillMean2(att):
    att = att.replace(np.NaN,1/att.mean())
    att = (att - att.mean()) / att.std()
    return att


#fill all NaN to the prob of unique values in att
def fillUnique(att):
    nullAtt = att.isnull().values
    u, c = np.unique(att.values[~nullAtt],return_counts = 1)
    att[nullAtt] = np.random.choice(u, nullAtt.sum(), p = c/c.sum())
    return att


#fill all NaN to the hard coded of values in att
def fillUnique2(att,u,c):
    nullAtt = att.isnull().values
    att[nullAtt] = np.random.choice(u, nullAtt.sum(), p = c/c.sum())
    att = (att - att.mean()) / att.std()
    return att


def result_to_csv(preds,knn_preds):
    Class = []
    Id = []
    count = 1
    for p in range(len(preds)):
        x = 0
        if knn_preds[p] == 1:
            x = preds[p][1]*1.2
        else:
            x = preds[p][1]
        Class.append(x)
        Id.append(count)
        count = count + 1

    d = {'Id': Id, 'Class': Class, }
    df = pd.DataFrame(data=d)
    df.to_csv('MLwork1Result.csv', index = False)


def my_norm(att):
    att = (att - att.mean()) / att.std()
    return att
np.random.seed(2018)

X_train.D_L_Age = fillMedian(X_train.D_L_Age)
X_test.D_L_Age = fillMedian(X_test.D_L_Age)
X_train.D_L_From_this_Treat_until_last_Optalm = fillMedian(X_train.D_L_From_this_Treat_until_last_Optalm)
X_test.D_L_From_this_Treat_until_last_Optalm = fillMedian(X_test.D_L_From_this_Treat_until_last_Optalm)
X_train.T_L_Treatment_Param_Cyl = fillMedian(X_train.T_L_Treatment_Param_Cyl)
X_test.T_L_Treatment_Param_Cyl = fillMedian(X_test.T_L_Treatment_Param_Cyl)
X_train.T_L_Treatment_SE = fillMedian(X_train.T_L_Treatment_SE)
X_test.T_L_Treatment_SE = fillMedian(X_test.T_L_Treatment_SE)
X_train.Pre_L_K_Minimum = fillMean(X_train.Pre_L_K_Minimum)
X_train.Pre_L_K_Maximum = fillMean(X_train.Pre_L_K_Maximum)
X_train.Pre_L_Steep_Axis_max = fillMean(X_train.Pre_L_Steep_Axis_max)
X_train.Pre_L_Subjective_Sph = fillMean(X_train.Pre_L_Subjective_Sph)
X_train.T_L_Target_SE = fillMean(X_train.T_L_Target_SE)
X_train.T_L_Treatment_ZO = fillMean(X_train.T_L_Treatment_ZO)
X_train.T_L_Actual_AblDepth = fillMean(X_train.T_L_Actual_AblDepth)
X_train.T_L_Humidity = fillMean(X_train.T_L_Humidity)
X_train.T_L_Temp = fillMean(X_train.T_L_Temp)
X_train.Pre_L_Est_UCVA_ = fillMean(X_train.Pre_L_Est_UCVA_)
#X_train.Pre_L_Est_BCVA = fillMean(X_train.Pre_L_Est_BCVA)
X_test.Pre_L_K_Minimum = fillMean(X_test.Pre_L_K_Minimum)
X_test.Pre_L_K_Maximum = fillMean(X_test.Pre_L_K_Maximum)
X_test.Pre_L_Steep_Axis_max = fillMean(X_test.Pre_L_Steep_Axis_max)
X_test.Pre_L_Subjective_Sph = fillMean(X_test.Pre_L_Subjective_Sph)
X_test.T_L_Target_SE = fillMean(X_test.T_L_Target_SE)
X_test.T_L_Treatment_ZO = fillMean(X_test.T_L_Treatment_ZO)
X_test.T_L_Actual_AblDepth = fillMean(X_test.T_L_Actual_AblDepth)
X_test.T_L_Humidity = fillMean(X_test.T_L_Humidity)
X_test.T_L_Temp = fillMean(X_test.T_L_Temp)
X_test.Pre_L_Est_UCVA_ = fillMean(X_test.Pre_L_Est_UCVA_)
#X_test.Pre_L_Est_BCVA = fillMean(X_train.Pre_L_Est_BCVA)
X_train = X_train.drop(['D_L_Dominant_Eye', 'Pre_L_Pupil_Day', 'Pre_L_Pupil_Night', 'Pre_L_Contact_Lens', 'Pre_L_Free_of_CL', 'Pre_L_Cycloplegia_Sph', 'Pre_L_Cycloplegia_Cyl', 'Pre_L_Cycloplegia_Axis', 'T_L_Year', 'T_L_Stop', 'T_L_Head', 'T_L_PTK_mm', 'T_L_PTK_mmm', 'T_L_MZ%', 'T_L_Epith._Rep.'], axis=1)
X_test = X_test.drop(['D_L_Dominant_Eye', 'Pre_L_Pupil_Day', 'Pre_L_Pupil_Night', 'Pre_L_Contact_Lens', 'Pre_L_Free_of_CL', 'Pre_L_Cycloplegia_Sph', 'Pre_L_Cycloplegia_Cyl', 'Pre_L_Cycloplegia_Axis', 'T_L_Year', 'T_L_Stop', 'T_L_Head', 'T_L_PTK_mm', 'T_L_PTK_mmm', 'T_L_MZ%', 'T_L_Epith._Rep.'], axis=1)

#X_train = X_train.drop(['D_L_Dominant_Eye','Pre_L_Pupil_Night', 'Pre_L_Contact_Lens', 'T_L_Year'], axis=1)
#X_test = X_test.drop(['D_L_Dominant_Eye', 'Pre_L_Pupil_Night', 'Pre_L_Contact_Lens', 'T_L_Year'], axis=1)

#X_train = X_train.drop(['D_L_Dominant_Eye', 'Pre_L_Pupil_Night', 'Pre_L_Contact_Lens', 'T_L_Year'], axis=1)
#X_test = X_test.drop(['D_L_Dominant_Eye', 'Pre_L_Pupil_Night', 'Pre_L_Contact_Lens', 'T_L_Year'], axis=1)


X_train.Pre_L_Subjective_Cyl = fillMean2(X_train.Pre_L_Subjective_Cyl)
X_train.Pre_L_Spherical_Equivalence = fillMean2(X_train.Pre_L_Spherical_Equivalence)
X_test.Pre_L_Subjective_Cyl = fillMean2(X_test.Pre_L_Subjective_Cyl)
X_test.Pre_L_Spherical_Equivalence = fillMean2(X_test.Pre_L_Spherical_Equivalence)

X_train.D_L_Sex = fillUnique(X_train.D_L_Sex)
X_train.D_L_Eye = fillUnique(X_train.D_L_Eye)
X_train.T_L_Laser_Type = fillUnique(X_train.T_L_Laser_Type)
X_train.T_L_Treatment_Type = fillUnique(X_train.T_L_Treatment_Type)
X_train.T_L_Cust_Ablation = fillUnique(X_train.T_L_Cust_Ablation)
X_train.T_L_Micro = fillUnique(X_train.T_L_Micro)
X_train.T_L_Ring = fillUnique(X_train.T_L_Ring)
X_train.T_L_Alchohol = fillUnique(X_train.T_L_Alchohol)
X_train.T_L_MZ_sec = fillUnique(X_train.T_L_MZ_sec)
X_train.T_L_Therapeutic_Cont_L = fillUnique(X_train.T_L_Therapeutic_Cont_L)

X_train.T_L_Ring = my_norm(X_train.T_L_Ring)
X_train.T_L_MZ_sec = my_norm(X_train.T_L_MZ_sec)

X_test.D_L_Sex = fillUnique(X_test.D_L_Sex)
X_test.D_L_Eye = fillUnique(X_test.D_L_Eye)
X_test.T_L_Laser_Type = fillUnique(X_test.T_L_Laser_Type)
X_test.T_L_Treatment_Type = fillUnique(X_test.T_L_Treatment_Type)
X_test.T_L_Cust_Ablation = fillUnique(X_test.T_L_Cust_Ablation)
X_test.T_L_Micro = fillUnique(X_test.T_L_Micro)
X_test.T_L_Ring = fillUnique(X_test.T_L_Ring)
X_test.T_L_Alchohol = fillUnique(X_test.T_L_Alchohol)
X_test.T_L_MZ_sec = fillUnique(X_test.T_L_MZ_sec)
X_test.T_L_Therapeutic_Cont_L = fillUnique(X_test.T_L_Therapeutic_Cont_L)

X_test.T_L_Ring = my_norm(X_test.T_L_Ring)
X_test.T_L_MZ_sec = my_norm(X_test.T_L_MZ_sec)

u = np.array([7,350000])
c = np.array([0.25,0.75])
X_train.Pre_L_Pachymetry = fillUnique2(X_train.Pre_L_Pachymetry, u, c)
u = np.array([45,350000])
c = np.array([0.5,0.5])
X_train.Pre_L_Average_K = fillUnique2(X_train.Pre_L_Average_K,u,c)
u = np.array([100,0])
c = np.array([0.25,0.75])
X_train.Pre_L_Subjective_Axis = fillUnique2(X_train.Pre_L_Subjective_Axis,u,c)
u = np.array([0,2000])
c = np.array([0.5,0.5])
X_train.T_L_Treatment_Param_Sph = fillUnique2(X_train.T_L_Treatment_Param_Sph,u,c)
u = np.array([0,120])
c = np.array([0.5,0.5])
X_train.T_L_Treatment_Param_Axis = fillUnique2(X_train.T_L_Treatment_Param_Axis,u,c)
u = np.array([0,120])
c = np.array([0.5,0.5])
X_train.T_L_Opt_Zo = fillUnique2(X_train.T_L_Opt_Zo,u,c)
u = np.array([0,80])
c = np.array([0.5,0.5])
X_train.T_L_Max_Abl_Depth = fillUnique2(X_train.T_L_Max_Abl_Depth,u,c)
u = np.array([25,38])
c = np.array([0.5,0.5])
X_train.T_L_Op_Time = fillUnique2(X_train.T_L_Op_Time,u,c)
u = np.array([7,350000])
c = np.array([0.25,0.75])
X_test.Pre_L_Pachymetry = fillUnique2(X_test.Pre_L_Pachymetry, u, c)
u = np.array([45,350000])
c = np.array([0.5,0.5])
X_test.Pre_L_Average_K = fillUnique2(X_test.Pre_L_Average_K,u,c)
u = np.array([100,0])
c = np.array([0.25,0.75])
X_test.Pre_L_Subjective_Axis = fillUnique2(X_test.Pre_L_Subjective_Axis,u,c)
u = np.array([0,2000])
c = np.array([0.5,0.5])
X_test.T_L_Treatment_Param_Sph = fillUnique2(X_test.T_L_Treatment_Param_Sph,u,c)
u = np.array([0,120])
c = np.array([0.5,0.5])
X_test.T_L_Treatment_Param_Axis = fillUnique2(X_test.T_L_Treatment_Param_Axis,u,c)
u = np.array([0,120])
c = np.array([0.5,0.5])
X_test.T_L_Opt_Zo = fillUnique2(X_test.T_L_Opt_Zo,u,c)
u = np.array([0,80])
c = np.array([0.5,0.5])
X_test.T_L_Max_Abl_Depth = fillUnique2(X_test.T_L_Max_Abl_Depth,u,c)
u = np.array([25,38])
c = np.array([0.5,0.5])
X_test.T_L_Op_Time = fillUnique2(X_test.T_L_Op_Time,u,c)
X_train.Pre_L_Steep_Axis_min = fillUnique(X_train.Pre_L_Steep_Axis_min)
X_test.Pre_L_Steep_Axis_min = fillUnique(X_test.Pre_L_Steep_Axis_min)

X_train.Pre_L_Steep_Axis_min = my_norm(X_train.Pre_L_Steep_Axis_min)
X_test.Pre_L_Steep_Axis_min = my_norm(X_test.Pre_L_Steep_Axis_min)

X_train = pd.get_dummies(X_train, columns = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Cust_Ablation","T_L_Micro","T_L_Alchohol","T_L_Therapeutic_Cont_L"], prefix = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Cust_Ablation","T_L_Micro","T_L_Alchohol","T_L_Therapeutic_Cont_L"])
X_test = pd.get_dummies(X_test, columns = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Cust_Ablation","T_L_Micro","T_L_Alchohol","T_L_Therapeutic_Cont_L"], prefix = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Cust_Ablation","T_L_Micro","T_L_Alchohol","T_L_Therapeutic_Cont_L"])

#X_train = pd.get_dummies(X_train, columns = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Therapeutic_Cont_L"], prefix = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Therapeutic_Cont_L"])
#X_test = pd.get_dummies(X_test, columns = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Therapeutic_Cont_L"], prefix = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Therapeutic_Cont_L"])


#X_train = pd.get_dummies(X_train, columns = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Therapeutic_Cont_L"], prefix = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Therapeutic_Cont_L"])
#X_test = pd.get_dummies(X_test, columns = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Therapeutic_Cont_L"], prefix = ["D_L_Sex","D_L_Eye","T_L_Laser_Type","T_L_Treatment_Type","T_L_Therapeutic_Cont_L"])


X_train, X_test = X_train.align(X_test, join= 'inner',axis=1)

Train = pd.concat([X_train, y_train], axis=1, sort=False)

Train0 = Train[Train.Class == 0]
Train1 = Train[Train.Class == 1]

Train0_down = resample(Train0,replace=False,n_samples=Train1.shape[0]*10,random_state=123)
new_Train = pd.concat([Train1,Train0_down])

#Train1_up = resample(Train1,replace=True,n_samples=3000,random_state=2018)
#new_Train = pd.concat([Train0,Train1_up])


X_train = new_Train.iloc[:, :-1]
y_train = new_Train.iloc[:, -1:]

print(X_train[y_train.Class == 1].shape[0])
print(X_train[y_train.Class == 0].shape[0])

clf = KNeighborsClassifier(n_neighbors=1,algorithm='auto')
clf.fit(X_train,y_train)
knn_preds = clf.predict(X_test)
print(knn_preds)
#knn_preds1 = knn_preds[knn_preds==1]
#print(knn_preds1)


np.random.seed(2018)
print(X_train.shape,y_train.shape)
xgb_class = xgb.XGBClassifier(max_depth=3,n_estimators=120,objective='binary:logistic',min_child_weight=2,gamma=0,learning_rate=0.02,colsample_bytree=0.8,reg_alpha=0.1,subsample=0.9,colsample_bylevel=0.6,reg_lambda=0.01,max_delta_step=2)
xgb_class.fit(X_train, y_train)
preds = xgb_class.predict_proba(X_test)
result_to_csv(preds,knn_preds)
#print(xgb_class.base_score)


param_test1 = {
# 'max_depth':[2,3,5],
# 'min_child_weight':[2,3,5],
# 'gamma':[0,0.2,0.4,0.6,0.8],
# 'subsample':[i/10.0 for i in range(6,10)],
# 'colsample_bytree':[i/10.0 for i in range(6,10)],
# 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
 #'learning_rate':[0.001,0.005,0.01,0.02,0.05,0.1]
    #'colsample_bylevel':[i/10.0 for i in range(6,10)],
    #'n_estimators':[100,120,140]
    #'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100],
    #'max_delta_step':[0,1,2,3,4,5],
    'scale_pos_weight':[1,0.09091,11]
}
#gsearch1 = GridSearchCV(estimator=xgb.XGBClassifier(max_depth=3,min_child_weight=2,gamma=0,learning_rate=0.02,colsample_bytree=0.8,reg_alpha=0.1,subsample=0.9,n_estimators=120,colsample_bylevel=0.6,reg_lambda=0.01,max_delta_step=2,
# objective='binary:logistic', nthread=4, seed=2018),
# param_grid=param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=10)
#gsearch1.fit(X_train, y_train)
#print(gsearch1.best_params_)



#Train = Train.iloc[:30451,:]
#print(Train.Class.value_counts())
#Train0 = Train[Train.Class == 0]
#Train1 = Train[Train.Class == 1]

#Train0_down = resample(Train0,replace=False,n_samples=Train1.shape[0]*11,random_state=123)
#new_Train = pd.concat([Train1,Train0_down])
#print(new_Train.Class.value_counts())

# Any results you write to the current directory are saved as output
