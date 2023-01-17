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
df_train = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/train.csv", header = None)

trainLabels = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/trainLabels.csv", header = None)

df_test = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/test.csv", header = None)
df_train.head()
df_test.head()
df_train.describe()
import statsmodels.api as sm

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,log_loss,f1_score



x=df_train

y=trainLabels.to_numpy()

x.head()
type(y)
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X=ss.fit_transform(x)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion='entropy',random_state=0,class_weight='balanced')



from sklearn.linear_model import LogisticRegression

LR=LogisticRegression(class_weight='balanced')



from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()



from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=100,random_state=0,class_weight='balanced')



from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()



from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier

gb=GradientBoostingClassifier(random_state=0)

bc=BaggingClassifier(base_estimator=knn,random_state=0)



import lightgbm as lgb

lgbm=lgb.LGBMClassifier(random_state=0)



from xgboost import XGBClassifier

classifier = XGBClassifier()
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint



rfc_tunned=RandomForestClassifier(n_estimators=100,random_state=0)

params={'n_estimators':sp_randint(1,1000),

        'max_features':sp_randint(1,40),

        'max_depth': sp_randint(2,50),

        'min_samples_split':sp_randint(2,80),

        'min_samples_leaf':sp_randint(1,80),

        'criterion':['gini','entropy']}



rsearch_rfc=RandomizedSearchCV(rfc_tunned,params,cv=3,scoring='accuracy',n_jobs=-1,random_state=0)



rsearch_rfc.fit(X,y)
rsearch_rfc.best_params_
rfc_tunned=RandomForestClassifier(**rsearch_rfc.best_params_,random_state=0)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

from scipy.stats import randint as sp_randint



knn_tunned=KNeighborsClassifier()



params={'n_neighbors':sp_randint(1,20),'p':sp_randint(1,30)}



rsearch_knn=RandomizedSearchCV(knn_tunned,params,cv=3,scoring='accuracy',n_jobs=-1,random_state=0)

rsearch_knn.fit(X,y)
rsearch_knn.best_params_
knn_tunned=KNeighborsClassifier(**rsearch_knn.best_params_)
from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform 



lgbm_tunned=lgb.LGBMClassifier(random_state=0)

params={'n_estimators':sp_randint(1,1000),

       'max_depth': sp_randint(2,80),

        'learning_rate':sp_uniform(0.001,0.05),

        'num_leaves':sp_randint(2,50)

       }



rsearch_lgbm=RandomizedSearchCV(lgbm_tunned,param_distributions=params,cv=3,scoring='accuracy',n_iter=200,n_jobs=-1,random_state=0)



rsearch_lgbm.fit(X,y)
rsearch_lgbm.best_params_
lgbm_tunned=lgb.LGBMClassifier(**rsearch_lgbm.best_params_,random_state=0)
from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform 



gb_tuned=GradientBoostingClassifier(random_state=0)

params= {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 

         'n_estimators':sp_randint(2,1500),

         'max_depth':sp_randint(1,10),

        'min_samples_split':sp_randint(2,100), 

         'min_samples_leaf':sp_randint(1,10),

        'max_features':sp_randint(1,9),

        'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]}



rsearch_gb=RandomizedSearchCV(gb_tuned,param_distributions=params,cv=3,n_iter=200,n_jobs=-1,random_state=0)



rsearch_gb.fit(X,y)
rsearch_gb.best_params_
gb_tuned=GradientBoostingClassifier(**rsearch_gb.best_params_,random_state=0)
models=[]

models.append(('Logistic',LR))

models.append(('Decision Tree',dt))

models.append(('Naive Bayes',nb))

models.append(('Random Forest',rfc))

models.append(('Random Forest Tunned',rfc_tunned))

models.append(('KNN',knn))

models.append(('KNN Tunned',knn_tunned))

models.append(('Bagging',bc))

models.append(('Gradient Boost',gb))

models.append(('Gradient Boost Tunned',gb_tuned))

models.append(('LGBM',lgbm))

models.append(('LGBM Tunned',lgbm_tunned))

models.append(('XGB',classifier))
from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.model_selection import train_test_split



results=[]

Var=[]

names=[]

for name,model in models:

    #kfold=model_selection.KFold(shuffle=True,n_splits=10,random_state=0)

    cv_results=cross_val_score(model,X,y,cv=10,scoring='roc_auc')

    results.append(np.mean(cv_results))

    Var.append(np.var(cv_results))

    names.append(name)



r_df=pd.DataFrame({'Model':names,'ROC-AUC':results,'Variance Error':Var})

print(r_df)
lgbm_tunned.fit(X,y)

y_pred2=lgbm_tunned.predict(df_test)
submission = pd.DataFrame(y_pred2)

print(submission.shape)

submission.columns = ['Solution']

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission
filename = 'output.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)
from subprocess import check_output

print(check_output(["ls", "../working"]).decode("utf8"))