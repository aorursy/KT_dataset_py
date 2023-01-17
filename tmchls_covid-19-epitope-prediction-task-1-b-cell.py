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
#reading the dataset for task 1
b=pd.read_csv('/kaggle/input/epitope-prediction/input_bcell.csv')
b.head()
b.columns
b.isnull().sum()
b['target'].value_counts()/len(b)*100.0
b.info()
#for statiscal analysis of continuous variables
b.describe()
#for statistical analysis of object variables
b.describe(include='all')
#to calculate peptide length
b['peptide_length']=b['end_position'] - b['start_position'] + 1
b.head()
#function to convert characters into their lengths
def length(col):
    for i in col:
        return len(i)
#converting all the three object type features
b['parent_protein_id']=length(b['parent_protein_id'])
b['protein_seq']=length(b['protein_seq'])
b['peptide_seq']=length(b['peptide_seq'])
x=b.drop(columns='target')
y=b['target']
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
import plotly.express as px#dropping unnecessary columns
b.drop(columns=['parent_protein_id','protein_seq','peptide_seq'],inplace=True)
plt.figure(figsize=(10,10))
sns.barplot(feature_importance_normalized,x.columns) 
plt.xlabel('Feature Labels') 
plt.ylabel('Feature Importances') 
plt.title('Comparison of different Feature Importances') 
plt.show() 
b.head()
b['peptide_length'].value_counts()/len(b)*100
features=["chou_fasman","emini","kolaskar_tongaonkar","parker","peptide_length","isoelectric_point","aromaticity",
            "hydrophobicity","stability"]
plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=2.0)
j=1
for i in features:
    plt.subplot(4,5,j)
    sns.distplot(b[i])
    j+=1
sns.set_style('whitegrid')
plt.figure(figsize=(30,30))
sns.catplot(y='isoelectric_point',x='peptide_length',data=b,ci=None,col='target',sharey=False)
plt.figure(figsize=(30,30))
sns.catplot(y='aromaticity',x='peptide_length',data=b,ci=None,col='target',sharey=False)
plt.figure(figsize=(30,30))
sns.catplot(y='hydrophobicity',x='peptide_length',data=b,ci=None,col='target',sharey=False)
plt.figure(figsize=(30,30))
sns.catplot(y='stability',x='peptide_length',data=b,ci=None,col='target',sharey=False)
x.head()
X=b.drop(columns='target')
Y=b['target']
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
l=LGBMClassifier(random_state=10)
l.fit(X_train,Y_train)
lg_train=l.predict(X_train)
roc_auc_score(lg_train,Y_train)
lg_pred=l.predict(X_valid)
lg_pred
roc_auc_score(lg_pred,Y_valid)
#predictions of validation dataset
predictions=pd.DataFrame(lg_pred,columns=['validation_pred'])
predictions.head()
#predicting on covid dataset
c=pd.read_csv('/kaggle/input/epitope-prediction/input_covid.csv')
c.head()
c.info()
c.drop(columns=['parent_protein_id','protein_seq','peptide_seq'],inplace=True)
c.head()
c.isnull().sum()
c['length']=c['end_position']-c['start_position'] + 1
d.transform(c)
y_pred=l.predict(c)
y_pred
y_pred=pd.DataFrame(y_pred,columns=['test_pred'])
y_pred.head()
y_pred.value_counts()/len(c)*100






































