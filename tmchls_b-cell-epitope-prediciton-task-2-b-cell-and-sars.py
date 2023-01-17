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
#loading the two datasets
b=pd.read_csv('/kaggle/input/epitope-prediction/input_bcell.csv')
s=pd.read_csv('/kaggle/input/epitope-prediction/input_sars.csv')
#combining the two datasets
c=pd.concat([b,s])
c.head()
c.info()
c.describe()
#for statistical analysis of object variables
c.describe(include='all')
c['target'].value_counts()/len(c)*100
#to calculate peptide length
c['peptide_length']=c['end_position'] - c['start_position'] + 1
#function to convert characters into their lengths
def length(col):
    for i in col:
        return len(i)
#converting all the three object type features
c['parent_protein_id']=length(c['parent_protein_id'])
c['protein_seq']=length(c['protein_seq'])
c['peptide_seq']=length(c['peptide_seq'])
x=c.drop(columns='target')
y=c['target']
#feature importance
from sklearn.ensemble import ExtraTreesClassifier
r = ExtraTreesClassifier(random_state=0)
r.fit(x,y)
feature_importance = r.feature_importances_
feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        r.estimators_], 
                                        axis = 0) 
#importing libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
plt.figure(figsize=(10,10))
sns.barplot(feature_importance_normalized,x.columns) 
plt.xlabel('Feature Labels') 
plt.ylabel('Feature Importances') 
plt.title('Comparison of different Feature Importances') 
plt.show() 
#dropping unnecessary columns
c.drop(columns=['parent_protein_id','protein_seq','peptide_seq'],inplace=True)
c.head()
c['peptide_length'].value_counts()/len(c)*100
features=["chou_fasman","emini","kolaskar_tongaonkar","parker","peptide_length","isoelectric_point","aromaticity",
            "hydrophobicity","stability"]
plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=2.0)
j=1
for i in features:
    plt.subplot(4,5,j)
    sns.distplot(c[i])
    j+=1
X=c.drop(columns='target')
Y=c['target']
#train and test
from sklearn.model_selection import train_test_split, RandomizedSearchCV
X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,stratify=Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import MinMaxScaler
d=MinMaxScaler()
d.fit_transform(X_train,Y_train)
d.transform(X_valid)
#fitting the lightbgm model 
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
params ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
l=LGBMClassifier()
w=RandomizedSearchCV(l,param_distributions=params,n_jobs=-1,cv=5,scoring='roc_auc')
w.fit(X_train,Y_train)
lg_pred=w.predict(X_valid)
lg_pred
roc_auc_score(lg_pred,Y_valid)
lg_train=w.predict(X_train)
roc_auc_score(lg_train,Y_train)
#predictions of validation dataset
predictions=pd.DataFrame(lg_pred,columns=['validation_pred'])
predictions.head()
predictions.value_counts()/len(c)*100
#predicting on covid dataset
co=pd.read_csv('/kaggle/input/epitope-prediction/input_covid.csv')
co.head()
co.info()
co.drop(columns=['parent_protein_id','protein_seq','peptide_seq'],inplace=True)
co.head()
co.isnull().sum()
co['length']=co['end_position']-co['start_position'] + 1
d.transform(co)
y_pred=w.predict(co)
y_pred
y_pred=pd.DataFrame(y_pred,columns=['test_pred'])
y_pred.head()
y_pred.value_counts()/len(co)*100