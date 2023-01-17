import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
df_train = pd.read_csv('../input/GiveMeSomeCredit/cs-training.csv')

df_test = pd.read_csv('../input/GiveMeSomeCredit/cs-test.csv')
print('Train Shape is :',df_train.shape,'\nTest shape is :',df_test.shape)
df_train.head(2)
(df_train.isna().sum()/len(df_train)) * 100
mask = df_train.isnull()

sns.heatmap(df_train, mask=mask,cmap="YlGnBu");
df_train.dropna(subset=['NumberOfDependents'],inplace=True)
df_train.shape
df_append = df_train.append(df_test)
(df_train.SeriousDlqin2yrs.value_counts() / len(df_train) ) * 100
sns.countplot(x="SeriousDlqin2yrs", data=df_train);
sns.distplot((df_train.age));
bins= [20,60,80,120]

labels_age = ['Adult','Young Senior','Senior']

df_append['AgeGroup'] = pd.cut(df_append['age'], bins=bins, labels=labels_age, right=False)

mask_2 = {

         'Adult':0,

         'Young Senior':1,

         'Senior':2}

df_append['AgeGroup'].replace(mask_2,inplace=True)
df_append['AgeGroup'].value_counts()
df_append['MonthlyIncome'].fillna(df_append['MonthlyIncome'].median(),inplace=True)

df_append['NumberOfDependents'].fillna(df_append['NumberOfDependents'].median(),inplace=True)
df_train = df_append[0:146076]

df_test = df_append[146076:]
#df_append = df_append[df_append != 5400]

df_adult = df_train[df_train['AgeGroup'] == 0]
sns.countplot(x='NumberOfDependents',data=df_adult);
sns.countplot(df_adult.SeriousDlqin2yrs);
sns.countplot(x="AgeGroup", data=df_train);
g = sns.jointplot("age", "NumberOfDependents", data=df_train, ylim=(0, 12),

                  color="m", height=7)
df_train['AgeGroup'].fillna(df_train['AgeGroup'].median(),inplace=True)
X = df_train.drop(columns={'Unnamed: 0','age','SeriousDlqin2yrs'})

y = df_train['SeriousDlqin2yrs']
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMModel,LGBMClassifier

from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import RFECV

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from imblearn.over_sampling import SMOTE

from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc,roc_curve

def plot_roc(pred):

    fpr,tpr,_ = roc_curve(y_test, pred)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10,8))

    plt.title('Receiver Operating Characteristic')

    sns.lineplot(fpr, tpr, label = 'AUC = %0.4f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print('Train Shapes',X_train.shape,' and ',y_train.shape,'\nTest Shapes',X_test.shape,' and ' ,y_test.shape,'\nOutput Values\n',y_train.value_counts())
smote = SMOTE(sampling_strategy = 'minority',k_neighbors = 2,random_state=0)

X_train_smote,y_train_smote = smote.fit_sample(X_train,y_train)



#Realizar Teste Smote + kfold(com cv)
print('Train Shapes',X_train_smote.shape,' and ',y_train_smote.shape,'\nTest Shapes',X_test.shape,' and ' ,y_test.shape,'\nOutput Values\n',y_train_smote.value_counts())
clf = RandomForestClassifier()

clf.fit(X_train,y_train)
feat_names = X.columns.values

importances = clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

indices = np.argsort(importances)[::-1][:20]



plt.figure(figsize=(12,12))

plt.title("Feature importances")

plt.bar(range(len(indices)), importances[indices], color="y", yerr=std[indices], align="center")

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')

plt.xlim([-1, len(indices)])

plt.show()
pred = clf.predict_proba(X_test)[:,1]

#Predict Primeiro modelo basico

print(roc_auc_score(y_test, pred))
plot_roc(pred)
Lgb = LGBMClassifier(objective='binary',metrics ='auc')
cv = StratifiedKFold(5)

visualizer = RFECV(Lgb, cv=cv, scoring='roc_auc')

visualizer.fit(X_train, y_train)        # Fit the data to the visualizer

visualizer.show();           # Finalize and render the figure
pred_rfe = visualizer.predict_proba(X_test)[:,1]
plot_roc(pred_rfe)
from hyperopt import hp, tpe, fmin

from sklearn.model_selection import cross_val_score
#Usando Hypteropt

space = {'n_estimators':hp.quniform('n_estimators', 10, 4000, 10),

        'learning_rate':hp.uniform('learning_rate', 0.00001, 0.03),

         'max_depth':hp.quniform('max_depth', 3,7,1),

         'subsample':hp.uniform('subsample', 0.60, 0.95),

         'colsample_bytree':hp.uniform('colsample_bytree', 0.60, 0.95),

         'reg_lambda': hp.uniform('reg_lambda', 1, 20),

        }



def objective(params):

    params = {'n_estimators': int(params['n_estimators']),

             'learning_rate': params['learning_rate'],

             'max_depth': int(params['max_depth']),

             'subsample': params['subsample'],

             'colsample_bytree': params['colsample_bytree'],

             'reg_lambda': params['reg_lambda'],

             }

    

    lgbm= LGBMClassifier(**params)

    cv = StratifiedKFold(5)

    score = cross_val_score(lgbm, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1).mean()

    return -score
best = fmin(fn= objective, space= space, max_evals=20, rstate=np.random.RandomState(1), algo=tpe.suggest)
lgbm = LGBMClassifier(random_state=0,

                        n_estimators=int(best['n_estimators']), 

                        colsample_bytree= best['colsample_bytree'],

                        learning_rate= best['learning_rate'],

                        max_depth= int(best['max_depth']),

                        subsample= best['subsample'],

                        reg_lambda= best['reg_lambda']

                       )



lgbm.fit(X_train, y_train)
lgbm_hype = lgbm.predict_proba(X_test)[:,1]
plot_roc(lgbm_hype)
#prediction

df_x_test = df_test.drop(columns={'Unnamed: 0','age','SeriousDlqin2yrs'})

pred = lgbm.predict_proba(df_x_test)[:,1]

#output

output = pd.DataFrame({'Id': df_test['Unnamed: 0'],'Probability': pred})

output.to_csv('submission.csv', index=False)