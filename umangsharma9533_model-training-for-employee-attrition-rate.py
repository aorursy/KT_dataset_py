# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import seaborn as sns 
sns.set()
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/Train.csv')
pd.options.display.max_rows=None
pd.options.display.max_columns=None
df_filled_NaN=df.fillna(df.median())
df_dummies=pd.get_dummies(df_filled_NaN['Compensation_and_Benefits'])
df_filled_NaN=df_filled_NaN.join(df_dummies)
df_filled_NaN['Gender']=df_filled_NaN['Gender'].map({'F':1,'M':0})
df_dummies_marrital=pd.get_dummies(df_filled_NaN['Relationship_Status'])
df_filled_NaN=df_filled_NaN.join(df_dummies_marrital)
df_filled_NaN_drop=df_filled_NaN.drop('Compensation_and_Benefits',axis=1)
df_filled_NaN_drop=df_filled_NaN_drop.drop('Relationship_Status',axis=1)
df_dummies_home=pd.get_dummies(df_filled_NaN_drop['Hometown'])
df_dummies_unit=pd.get_dummies(df_filled_NaN_drop['Unit'])
df_dummies_decision=pd.get_dummies(df_filled_NaN_drop['Decision_skill_possess'])
df_filled_NaN_drop_drop=df_filled_NaN_drop.join([df_dummies_home,df_dummies_unit,df_dummies_decision])
df_filled_NaN_drop_drop_final=df_filled_NaN_drop_drop.drop(['Hometown','Unit','Decision_skill_possess'],axis=1)

df_filled_NaN_drop_drop_final

X=df_filled_NaN_drop_drop_final[['Gender','Time_of_service','Post_Level','Pay_Scale','Work_Life_balance','VAR2','IT','Behavioral','Conceptual']]
#X=df_filled_NaN_drop_drop_final[['Gender','Age','Time_of_service','Time_since_promotion','growth_rate','Post_Level','Pay_Scale','Work_Life_balance','VAR2','VAR7','type0','type3','Logistics','IT','Springfield','Operarions','Purchasing','R&D','Behavioral','Conceptual','Directive']]
#X=df_filled_NaN_drop_drop_final[['Post_Level','Work_Life_balance','type0','Conceptual']]
#X=df_filled_NaN_drop_drop_final[['Work_Life_balance','type0']]
Y=df_filled_NaN_drop_drop_final['Attrition_rate']
import xgboost as xgb
model = xgb.XGBRegressor()

parameters = {'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30], #so called `eta` value
              'max_depth': [ 3, 4, 5, 6, 8, 10, 12, 15],
              'min_child_weight': [ 1, 3, 5, 7],
              "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]}

xgb_grid = GridSearchCV(model,
                        parameters,
                        n_jobs=5,cv=5,verbose=3)

xgb_grid.fit(X,
         Y)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
xgb_grid.best_estimator_
model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.3, gamma=0.4, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.2, max_delta_step=0, max_depth=10,
             min_child_weight=3, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
from sklearn.model_selection import cross_val_score
score=cross_val_score(model,X,Y,cv=10)
score

score.mean()
model.fit(X,Y)
df_test=pd.read_csv('/kaggle/input/Test.csv')
df_test=df_test.fillna(df_test.median())
df_test_dummies=pd.get_dummies(df_test['Compensation_and_Benefits'])
df_test=df_test.join(df_test_dummies)
df_test['Gender']=df_test['Gender'].map({'F':1,'M':0})
df_dummies_marital=pd.get_dummies(df_test['Relationship_Status'])
df_test=df_test.join(df_dummies_marital)
df_filled_NaN_test_drop=df_test.drop('Compensation_and_Benefits',axis=1)
df_filled_NaN_test_drop=df_filled_NaN_test_drop.drop('Relationship_Status',axis=1)
df_dummies_test_home=pd.get_dummies(df_filled_NaN_test_drop['Hometown'])
df_dummies_test_unit=pd.get_dummies(df_filled_NaN_test_drop['Unit'])
df_dummies_test_decision=pd.get_dummies(df_filled_NaN_test_drop['Decision_skill_possess'])
df_filled_NaN_test_drop_drop=df_filled_NaN_test_drop.join([df_dummies_test_home,df_dummies_test_unit,df_dummies_test_decision])
df_filled_NaN_drop_drop_test_final=df_filled_NaN_test_drop_drop.drop(['Hometown','Unit','Decision_skill_possess'],axis=1)

testX=df_filled_NaN_drop_drop_test_final[['Gender','Time_of_service','Post_Level','Pay_Scale','Work_Life_balance','VAR2','IT','Behavioral','Conceptual']]
#81.17testX=df_filled_NaN_drop_drop_test_final[['Gender','Age','Time_of_service','Time_since_promotion','growth_rate','Post_Level','Pay_Scale','Work_Life_balance','VAR2','VAR7','type0','type3','Logistics','IT','Springfield','Operarions','Purchasing','R&D','Behavioral','Conceptual','Directive']]
#81.26testX=df_filled_NaN_drop_drop_test_final[['Time_since_promotion','growth_rate','Post_Level','Work_Life_balance','type0','Springfield','Operarions','Purchasing','R&D','Conceptual','Directive']]
#testX=df_filled_NaN_drop_drop_test_final[['Post_Level','Work_Life_balance','type0','Conceptual']]
#testX=df_filled_NaN_drop_drop_test_final[['Work_Life_balance','type0']]
predictions=model.predict(testX)
df_final=pd.DataFrame(df_test['Employee_ID'])
predication_xg_df=pd.DataFrame(predictions)
df_final['Attrition_rate']=predication_xg_df
df_final.to_csv('submission.csv',index=False)