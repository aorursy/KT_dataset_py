import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold,KFold
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix,accuracy_score,f1_score

import optuna
from optuna.samplers import TPESampler
train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
sample = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv')

sns.set(style='white', context='notebook', palette='deep')
train.head()
train.info()
train.describe()
train.isnull().sum()
train.skew()
train.dtypes
all_features = pd.concat([train.drop(['id','Response'],axis=1),test.drop('id',axis=1)],axis=0)
y = train['Response']
fig = px.pie(train,values=train['Response'].value_counts(),names=['Class 0','Class 1'],hole=0.6,labels={0:'Response = 0'},color_discrete_sequence=px.colors.sequential.Sunset)
fig.show(showlegend=True)
sns.countplot(train['Response'])
plt.show()
sns.barplot(train['Driving_License'],train['Response'])
plt.show()
sns.barplot(train['Previously_Insured'],train['Response'])
plt.show()
sns.barplot(train['Vehicle_Damage'],train['Response'])
plt.show()
sns.barplot(train['Vehicle_Age'],train['Response'])
plt.show()
sns.barplot(train['Gender'],train['Response'])
plt.show()
plt.figure(figsize=(20,10))
sns.barplot(train['Age'],train['Response'])
plt.show()
sns.boxplot(train['Age'])
plt.show()
bins = [20, 30, 40, 50, 60, 70,90]
labels = ['20-27', '28-39', '40-49', '50-59', '60-69', '70+']
age_categories = pd.cut(train['Age'], bins, labels = labels,include_lowest = True)
sns.barplot(age_categories,train['Response'])
plt.show()
sns.boxplot(train['Annual_Premium'])
plt.show()
g = sns.distplot(train['Annual_Premium'],label='Skewness: '+str(round(train['Annual_Premium'].skew(),4)))
g = g.legend(loc='best')
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(train.corr(),annot=True,cmap='rainbow')
plt.show()
all_features['Vehicle_Age'] = all_features['Vehicle_Age'].map({'> 2 Years':2,'1-2 Year':1,'< 1 Year':0})
all_features['Vehicle_Damage'] = all_features['Vehicle_Damage'].map({'Yes':1,'No':0})
all_features['Gender'] = all_features['Gender'].map({'Male':1,'Female':0}) 
all_features
X = all_features.iloc[:len(train),:]
X_test = all_features.iloc[len(train):,:]

kf = StratifiedKFold(n_splits=12,shuffle=True,random_state=42)
for train_index,val_index in kf.split(X,y):
    X_train,X_val = X.iloc[train_index],X.iloc[val_index],
    y_train,y_val = y.iloc[train_index],y.iloc[val_index],
rus = RandomOverSampler(random_state=42)
X_rus,y_rus = rus.fit_sample(X_train,y_train)
lgb_rus = LGBMClassifier(random_state=42)
lgb_rus.fit(X_rus,y_rus)
print(classification_report(y_val,lgb_rus.predict(X_val)))
print('ROC AUC Score: ' + str(roc_auc_score(y_val,lgb_rus.predict(X_val))))
sns.heatmap(confusion_matrix(y_val,lgb_rus.predict(X_val)),cmap='magma',annot=True,fmt='g')
plt.show()
def create_model(trial):
    n_estimators = trial.suggest_int('n_estimators',100,500)
    num_leaves = trial.suggest_int('num_leaves',10,500)
    max_depth = trial.suggest_int('max_depth',4,20)
    learning_rate = trial.suggest_uniform('learning_rate',0.0001,1)
    min_child_samples = trial.suggest_int('min_child_samples',10,50)
    model = LGBMClassifier(n_estimators=n_estimators,num_leaves=num_leaves,
    max_depth=max_depth,learning_rate=learning_rate,min_child_samples=min_child_samples)
    return model

def objective(trial):
    model = create_model(trial)
    model.fit(X_rus,y_rus)
    score = roc_auc_score(y_val,model.predict(X_val))
    return score

sampler = TPESampler(seed=42)
study = optuna.create_study(sampler=sampler,direction='maximize')
study.optimize(objective,n_trials=60)
lgb_params = study.best_params
lgb_params['random_state'] = 42
lgb = LGBMClassifier(**lgb_params)
lgb.fit(X_rus, y_rus)
preds = lgb.predict(X_val)
print(classification_report(y_val,lgb.predict(X_val)))
print('ROC AUC Score: ' + str(roc_auc_score(y_val,lgb.predict(X_val))))
sns.heatmap(confusion_matrix(y_val,lgb.predict(X_val)),cmap='magma',annot=True,fmt='g')
plt.show()