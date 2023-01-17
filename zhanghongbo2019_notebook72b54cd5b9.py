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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import *
from math import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
df=pd.read_csv("../input/changdaojunqun/otu.tsv",sep="\t")
sample_list=list(df.groupby(df["sample"]))

def df_trans(my_tumple):
    return pd.DataFrame(my_tumple[1][['reads']].values.T,index=[my_tumple[0]],columns=my_tumple[1]['otu'].tolist())
otu=pd.concat(map(df_trans,sample_list))
otu.fillna(0, inplace=True)
otu=otu.astype(int)
#otu_filter = otu.loc[:, (otu.sum(axis=0) > 100)]
label=pd.read_csv("../input/changdaojunqun/label.tsv",sep="\t",index_col="sample")
#otu_label=pd.concat([label, otu_filter], axis=1)
otu_label=pd.concat([label, otu], axis=1)
otu_label
otu_label.columns
#train
X=otu_label.drop(['label'], axis=1)# Features
y=otu_label['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
X_train.shape
import xgboost as xgb
param_dict = {'n_estimators':range(50,500,10),
              'max_depth':range(2,15,1),
              'learning_rate':np.linspace(0.01,2,20),
              'subsample':np.linspace(0.7,0.9,20),
              'colsample_bytree':np.linspace(0.5,0.98,10),
              'min_child_weight':range(1,9,1)
             }
gsearch2 = GridSearchCV(estimator=xgb.XGBClassifier(),
                        param_grid=param_dict,
                        n_jobs=-1,
                        cv=10)
#在训练集上训练
gsearch2.fit(X_train,y_train)
print(f'best params:{gsearch2.best_params_}')
# 返回准确率
print('best accuracy:%f' % gsearch2.best_score_)
# 返回最佳训练器
best_estimator = gsearch2.best_estimator_
print(best_estimator)
#对训练集进行交叉验证
scores = cross_val_score(best_estimator, X_train, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#对测试集测试
best_estimator.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#混淆矩阵
print(classification_report(y_test, y_pred))