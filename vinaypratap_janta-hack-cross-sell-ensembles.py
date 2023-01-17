import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
import seaborn as sns

# to see all the comands result in a single kernal 
%load_ext autoreload
%autoreload 2
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/crosssell/train.csv")
test = pd.read_csv("/kaggle/input/crosssell/test.csv")
train.shape
test.shape
train.info()
test.info()
df = pd.concat([train,test])
df.head()
df.shape
df.isnull().sum()
#converting object to int type
df['Vehicle_Age']=df['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})
df['Gender']=df['Gender'].replace({'Male':1,'Female':0})
df['Vehicle_Damage']=df['Vehicle_Damage'].replace({'Yes':1,'No':0})

df.head(10)
df.columns
df['Age_Mean'] = df.groupby(['Region_Code'])['Age'].mean()
df.groupby('Age')['Previously_Insured'].count()
df['Age'].describe()
# #df['Age_Band'] = df['Age'].replace({'Yes':1,'No':0})
# ### handling age columns and creating brackets

# df.loc[(df['Age']<=25),'Age'] = 0
# df.loc[(df['Age']>25) & (df['Age']<=36) ,'Age'] = 1
# df.loc[(df['Age']>36) & (df['Age']<=50) ,'Age'] = 2
# df.loc[(df['Age']>50),'Age'] = 3
df['Vintage'] = df['Vintage'] / 365
df.info()
col1 = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
cat_cols = ['Gender','Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel']
#df.drop('Age',axis=1,inplace=True)
# changing data type because cat_feature in catboost cannot be float
df['Region_Code']=df['Region_Code'].astype(int)

df['Policy_Sales_Channel']=df['Policy_Sales_Channel'].astype(int)

df.head()
train = df[:381109]
test = df[381109:]
test = test.drop('Response',axis=1)
X = train[col1]
y = train['Response']
# lets start model building
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=150303,stratify=y,shuffle=True)
!pip install catboost
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
catb = CatBoostClassifier(boosting_type='Ordered',max_depth=8)
catb = catb.fit(X_train,y_train,cat_features = cat_cols,eval_set = (X_test,y_test),plot=True,early_stopping_rounds=50,verbose=100)
y_cat = catb.predict(X_test)
pred_cat= catb.predict_proba(test[col1])[:, 1]   ##predict on test data
probs_cat_train = catb.predict_proba(X_train)[:,1]
probs_cat_test = catb.predict_proba(X_test)[:,1]
roc_auc_score(y_train,probs_cat_train)
roc_auc_score(y_test,probs_cat_test)
feat_importances = pd.Series(catb.feature_importances_, index=X_train.columns)
feat_importances.nlargest(15).plot(kind='barh')
#feat_importances.nsmallest(20).plot(kind='barh')
plt.show()
# Training on full train data
from lightgbm import LGBMClassifier

LGB=LGBMClassifier(boosting_type='gbdt', max_depth=8, learning_rate=0.01, objective='binary',
                  reg_lambda=8, n_jobs=-1, n_estimators=1243,reg_alpha=2.2)
                 

model = LGB.fit(X,y)

pred_lgb = model.predict_proba(test[col1])[:, 1]

# Training on full train data
from xgboost import XGBClassifier

#XGB=XGBClassifier(n_estimators=1243,learning_rate=0.01,reg_lambda=8)

XGB = XGBClassifier(learning_rate=0.01, max_depth=8,
                                                          missing=None, n_estimators=1450, nthread=15,
                                                           objective='binary:logistic', reg_alpha=2.2, reg_lambda=8,
                                                      scale_pos_weight=2.5,  silent=True, subsample=.7)   #.7 best
                 
## estimators 1500 is best till now with these parms
model_xgb = XGB.fit(X,y)

pred_xgb = model_xgb.predict_proba(test[col1])[:, 1]
mix_pred = (pred_cat*0.7 + pred_lgb*0.1 + pred_xgb*0.2)
#mix_pred = (pred_cat*0.8 + pred_xgb*0.2)
#mix_pred = (pred_xgb*1)
sample_submmission = pd.read_csv("/kaggle/input/crosssell/sample_submission.csv")
sample_submmission['Response']=mix_pred
sample_submmission.to_csv("cat.csv", index = False)