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
df=pd.read_csv("/kaggle/input/pbs-eda/data_with_label")
df.head()
df.columns
df=df[df['world']=="'TREETOPCITY'"]
df.head()
df=df.drop(['world','type','timestamp','game_session','event_id','Unnamed: 0','installation_id','event_code'],axis=1)
df.head()
import ast
import statistics
def game_time(df1):
    game_avg_time=[]
    for i in range(df1.shape[0]):
        game_avg_time.append(statistics.mean(ast.literal_eval(df1.iloc[i])))
    return game_avg_time
def min_max(df1):
    game_min_time=[]
    game_max_time=[]
    for i in range(df1.shape[0]):
        temp=ast.literal_eval(df1.iloc[i])
        game_min_time.append(min(temp))
        game_max_time.append(max(temp))
    return  game_min_time,game_max_time
df['game_avg_time']=game_time(df['game_time'])
df['event_count_avg']=game_time(df['event_count'])
df['game_start'],df['game_end']=min_max(df['game_time'])
df['event_count_min'],df['event_count_max']=min_max(df['event_count'])
df.head()
df=df.drop(['event_data','event_count','game_time'],axis=1)
from sklearn.preprocessing import MinMaxScaler
def normalize(X):
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X)
    return X_train_minmax
for i in ['num_correct','num_incorrect','game_avg_time','event_count_avg', 'game_start', 'game_end', 'event_count_min','event_count_max']:
    df[i]=normalize(df[i].values.reshape(-1, 1))
    df[i]=normalize(df[i].values.reshape(-1, 1))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['accuracy_group'],axis=1),df['accuracy_group'],test_size=0.33, stratify=df['accuracy_group'])
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
ohe=ohe.fit(X_train.title.values.reshape(-1,1))
ohe_train=ohe.transform(X_train.title.values.reshape(-1,1)).toarray()
ohe_test=ohe.transform(X_test.title.values.reshape(-1,1)).toarray()
ohe_train_df=pd.DataFrame(ohe_train,columns=['Mushroom Sorter (Assessment)','Bird Measurer (Assessment)'])
ohe_test_df=pd.DataFrame(ohe_test,columns=['Mushroom Sorter (Assessment)','Bird Measurer (Assessment)'])
X_train=pd.concat([X_train,ohe_train_df],axis=1)
X_test=pd.concat([X_test,ohe_train_df],axis=1)
X_train=X_train.drop(['title'],axis=1)
X_test=X_test.drop(['title'],axis=1)
X_train.head()
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
mnb = MultinomialNB(class_prior = [0.5, 0.5])
param={'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000]}
clf=GridSearchCV(mnb,param,cv=3,scoring='roc_auc',return_train_score=True,n_jobs=-1)
clf.fit(X_train,y_train)
result=pd.DataFrame.from_dict(clf.cv_results_)
result=result.sort_values(['param_alpha'])
trainauc=result['mean_train_score']
trainaucstd=result['std_train_score']
cvauc=result['mean_test_score']
cvaucstd=result['std_test_score']
K=result['param_alpha']
for i in range(len(K)):
 K[i]=np.log10(K[i])
plt.plot(K,trainauc,label='Train AUC')
plt.plot(K,cvauc,label="CV AUC")
plt.scatter(K,trainauc,label='Train AUC Points')
plt.scatter(K,cvauc,label='CV AUC Points')
plt.legend()
plt.xlabel("Alpha: hyperparametr")
plt.ylabel("AUC")
plt.title("Hyper parameter Vs AUC plot (Bag of words)")
plt.grid()
plt.show()
result.head()