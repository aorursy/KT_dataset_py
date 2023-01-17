## import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/datasetshr/train.csv")
print(train.shape)
train.head()
test = pd.read_csv("/kaggle/input/datasetshr/test.csv")
print(test.shape)
test.head()
df = pd.concat([train,test])
print(df.shape)
df.head()
df.info()
## percentage of null values in every field
Null_per = pd.Series((df.isnull().sum()*100)/len(df))
Null_per = round(Null_per.sort_values(ascending=False),2)
Null_per
print(df['education'].unique())
print(df['previous_year_rating'].unique())
df[df['education'].isnull()]
df.groupby(['department','education'])['education'].count().plot(kind='bar')
#lets remove single quote from values in any field specially education
df = df.replace({'\'': ' '}, regex=True)
## seems bachelor's degree would be apt for missing value in education
df['education'] = df['education'].fillna("Bachelor s")
df.isnull().sum()
df[df['previous_year_rating'].isnull()]
## lets impute missing value in previous_year_rating 
df['previous_year_rating'] = df['previous_year_rating'].fillna(0)
df.isnull().sum()
df.groupby(['is_promoted','previous_year_rating','awards_won?'])['previous_year_rating'].count().plot(kind='bar')
df.groupby(['is_promoted','awards_won?'])['awards_won?'].count().plot(kind='bar')
df.groupby(['is_promoted','KPIs_met >80%'])['KPIs_met >80%'].count().plot(kind='bar')
df.groupby(['is_promoted','gender'])['gender'].count().plot(kind='bar')
df['gender'] = df['gender'].map({'m':1,'f':0})
df.groupby(['is_promoted','no_of_trainings'])['no_of_trainings'].count().plot(kind='bar')
df.groupby(['is_promoted','education'])['education'].count().plot(kind='bar')
df['education'] = df['education'].map({'Bachelor s':0,'Below Secondary':1,'Master s & above':2})
df.groupby(['is_promoted','department'])['department'].count().plot(kind='bar')
df['department'] = df['department'].map({'Analytics':0,'Finance':1,'HR':2,'Legal':3,'Operations':4,'Procurement':5,'R&D':6,'Sales & Marketing':7,'Technology':8})
df.groupby(['is_promoted','recruitment_channel'])['recruitment_channel'].count().plot(kind='bar')
df['recruitment_channel'] = df['recruitment_channel'].map({'other':0,'referred':1,'sourcing':2})
df['region'] = df['region'].str.replace("region_",'')
## binning of length of service columns
bins = [0,3,6,10,15,25,70]
labels = ['LessThan3','LessThan6','LessThan10','LessThan15','LessThan25','MoreThan25']
df['len_of_serv_bins'] = pd.cut(df['length_of_service'],bins=bins,labels=labels)
df.head()
df.groupby(['is_promoted','len_of_serv_bins'])['len_of_serv_bins'].count().plot(kind='bar')
## binning of age columns
bins = [15,25,30,35,45,100]
labels = ['LessThan25','LessThan30','LessThan35','LessThan45','MoreThan45']
df['age_bins'] = pd.cut(df['age'],bins=bins,labels=labels)
df.head()
df.groupby(['is_promoted','age_bins'])['age_bins'].count().plot(kind='bar')
df['age_bins'] = df['age_bins'].map({'LessThan25':0,'LessThan30':1,'LessThan35':2,'LessThan45':3,'MoreThan45':5})
df['len_of_serv_bins'] = df['len_of_serv_bins'].map({'LessThan3':0,'LessThan6':1,'LessThan10':2,'LessThan15':3,'LessThan25':4,'MoreThan25':5})
df['sum_metric'] = df['awards_won?']+df['KPIs_met >80%'] + df['previous_year_rating']
df['tot_score'] = df['avg_training_score'] * df['no_of_trainings']
df.drop(['age','length_of_service'],axis=1,inplace=True)
df.head()
df['region'] = df['region'].astype(int)
df['len_of_serv_bins'] = df['len_of_serv_bins'].astype(int)
df['age_bins'] = df['age_bins'].astype(int)
train = df[:54808]
test = df[54808:]
train.drop(['employee_id'],axis=1,inplace=True)
X = train.drop('is_promoted',axis=1)
y = train['is_promoted']
X.info()
test_c = test.copy()
test = test.drop(['is_promoted','employee_id'],axis=1)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(X)
# X_pca = pca.transform(X)
# test_pca = pca.transform(test)
# X=np.column_stack((X,X_pca))
# test=np.column_stack((test,test_pca))
# scale = StandardScaler()
# X = scale.fit_transform(X)
# test = scale.transform(test)
# #lets print stats after smote
# print("counts of label '1':",sum(y==1))
# print("counts of label '0':",sum(y==0))
# #perform oversampling using smote
# import six
# import sys
# import joblib
# sys.modules['sklearn.externals.six'] = six
# sys.modules['sklearn.externals.joblib'] = joblib
# from imblearn.over_sampling import SMOTE


# sm = SMOTE(random_state=1)
# X, y = sm.fit_sample(X, y)
# #lets print stats after smote
# print("counts of label '1':",sum(y==1))
# print("counts of label '0':",sum(y==0))
from sklearn.metrics import f1_score
rf = RandomForestClassifier(n_estimators=2000,max_depth=20,min_samples_split=5,max_features='sqrt')
rf.fit(X,y)
y_pred_rf = rf.predict(test).astype(int)
print("Accuracy_Score:",rf.score(X,y))
#print("F1_Score:",f1_score(y,y_pred_rf, average='weighted'))
#xgb = XGBClassifier(n_estimators=2000,learning_rate=0.1,reg_lambda=0.3,gamma=8,subsample=0.2)
#xgb = XGBClassifier(learning_rate =0.1, n_estimators=494, max_depth=5,subsample = 0.70, 
#                                                              scale_pos_weight = 2.5,updater ="grow_histmaker",base_score  = 0.2,)
xgb = XGBClassifier(base_score=0.5, gamma=0.3, learning_rate=0.1, max_delta_step=0, max_depth=5,
                                                            missing=None, n_estimators=494, nthread=15,
                                                           objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                                                           scale_pos_weight=2.5,  silent=True, subsample=1)


xgb.fit(X,y)
y_pred_xgb = xgb.predict(test).astype(int)
print("Accuracy_Score:",xgb.score(X,y))
#xgb = XGBClassifier(n_estimators=2000,learning_rate=0.1,reg_lambda=0.3,gamma=8,subsample=0.2)
xgb1 = XGBClassifier(learning_rate =0.1, n_estimators=494, max_depth=5,subsample = 0.70, 
                                                              scale_pos_weight = 2.5,updater ="grow_histmaker",base_score  = 0.2)

xgb1.fit(X,y)
y_pred_xgb1 = xgb1.predict(test).astype(int)
print("Accuracy_Score:",xgb1.score(X,y))
#lgb = LGBMClassifier(n_estimators=1000,learning_rate=0.095,reg_lambda=0.4,gamma=10,subsample=0.2)  ### .4911 f1 score highest
#lgb = LGBMClassifier(n_estimators=1000,learning_rate=0.095,reg_lambda=0.4,gamma=10,subsample=0.2,colsample_bytree = 0.3,min_child_weight=3)   ##.497297 f1 score
lgb = LGBMClassifier(subsample_freq = 2, objective ="binary",importance_type = "gain",verbosity = -1, max_bin = 60,num_leaves = 300,boosting_type = 'dart',learning_rate=0.15, n_estimators=494, max_depth=5, scale_pos_weight=2.5) 


lgb.fit(X,y)
y_pred_lgb = lgb.predict(test).astype(int)   
print("Accuracy_Score:",lgb.score(X,y))
#lgb = LGBMClassifier(n_estimators=1000,learning_rate=0.095,reg_lambda=0.4,gamma=10,subsample=0.2)  ### .4911 f1 score highest
#lgb = LGBMClassifier(n_estimators=1000,learning_rate=0.095,reg_lambda=0.4,gamma=10,subsample=0.2,colsample_bytree = 0.3,min_child_weight=3)   ##.497297 f1 score
lgb1 = LGBMClassifier(  bagging_fraction=0.9, feature_fraction=0.9, subsample_freq = 2, objective ="binary",importance_type = "gain",verbosity = -1, max_bin = 60,num_leaves = 300,boosting_type = 'dart',learning_rate=0.15, n_estimators=494, max_depth=5, scale_pos_weight=2.5)


lgb1.fit(X,y)
y_pred_lgb1 = lgb1.predict(test).astype(int)   
print("Accuracy_Score:",lgb1.score(X,y))
#catb = CatBoostClassifier(n_estimators=1000,learning_rate=0.095,random_state=2)
#catb = CatBoostClassifier(learning_rate=0.15, n_estimators=494, max_depth=5, scale_pos_weight=2.5,random_strength= None)
catb = CatBoostClassifier(learning_rate=0.15, n_estimators=494, max_depth=5, scale_pos_weight=2.5,
                                                random_strength= 0.157)                                                                                       
catb.fit(X,y)
y_pred_catb = catb.predict(test).astype(int)
print("Accuracy_Score:",catb.score(X,y))
#catb = CatBoostClassifier(n_estimators=1000,learning_rate=0.095,random_state=2)
catb1 = CatBoostClassifier(learning_rate=0.15, n_estimators=494, max_depth=5, scale_pos_weight=2.5,random_strength= None)
                                                                              
catb1.fit(X,y)
y_pred_catb1 = catb1.predict(test).astype(int)
print("Accuracy_Score:",catb1.score(X,y))
d=pd.DataFrame()
d=pd.concat([d,pd.DataFrame(catb1.predict(test).astype(int)),pd.DataFrame(xgb.predict(test).astype(int)),pd.DataFrame(lgb.predict(test).astype(int))],axis=1)
d.columns=['1','2','3']

re=d.mode(axis=1)[0]
re.head()
submission = pd.DataFrame({
        "employee_id": test_c["employee_id"],
        "is_promoted": re             ### prediting using ensemble model
    })

submission.to_csv('HR_submission.csv', index=False)



