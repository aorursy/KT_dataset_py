import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from xgboost import XGBClassifier



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



        

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import warnings

warnings.filterwarnings("ignore")





from sklearn.metrics import accuracy_score

df = pd.read_csv("/kaggle/input/ultimate-ufc-dataset/ufc-master.csv")  # The train dataset 

df_pred = pd.read_csv("/kaggle/input/ultimate-ufc-dataset/upcoming-event.csv") # The pred dataset 
df['train_data'] = 1 

df_pred['train_data'] = 0 

df = pd.concat([df,df_pred])
df.sample(5)
def get_winner(color) : return 1 if color =='Red' else 0
df['won'] = df.Winner.apply(lambda x: get_winner(x))
#Feature selection

features = ['R_odds','B_odds','R_ev','B_ev','title_bout','weight_class','no_of_rounds',\

           'B_current_lose_streak','B_current_win_streak',\

            'B_wins','B_losses',\

            'B_age','B_Stance','B_Height_cms','B_Reach_cms','B_Weight_lbs',\



            'R_wins','R_losses',\

            'R_current_lose_streak','R_current_win_streak',\

           'R_age','R_Stance','R_Height_cms','R_Reach_cms','R_Weight_lbs'

           ]



selected_columns = features + ["train_data" , "won"]
df=pd.get_dummies(df[selected_columns])
X = df[df['train_data'] == 1]

y = df['won'][df['train_data'] == 1]



X_pred = df[df['train_data'] == 0 ]
X = X.drop(['train_data','won'],axis = 1 )

X_pred = X_pred.drop(['train_data','won'],axis = 1 )
from sklearn.model_selection import GridSearchCV  

param_list = {

 'max_depth':range(1,6),

 'min_child_weight':range(1,2),

 'n_estimators': range(10,100,40)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.001, max_depth=5,\

                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,\

                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),\

                                    param_grid = param_list, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1 = gsearch1.fit(X,y)

from sklearn.model_selection import GridSearchCV  

param_test1 = {

 'max_depth':range(1,6),

 'min_child_weight':range(1,2),

 'n_estimators': range(10,100,40)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.001, max_depth=5,\

                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,\

                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),\

                                    param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=5)

gsearch1 = gsearch1.fit(X,y)
print(f"Best score = {gsearch1.best_score_}")

print(f"Best parameters = {gsearch1.best_params_ }")
final_xgb_model = XGBClassifier(learning_rate =0.001, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8,\

                                colsample_bytree=0.8,objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27 )

final_xgb_model= final_xgb_model.set_params(**gsearch1.best_params_)

final_xgb_model.fit(X ,y)
import matplotlib.pyplot as plt 

from xgboost import plot_importance

plot_importance(final_xgb_model)

plt.show();
final_prediction_proba =  pd.DataFrame(final_xgb_model.predict_proba(X_pred))

final_prediction_proba.columns = ['pred_0','pred_1']



predict_proba = pd.DataFrame({"R_fighter": df_pred['R_fighter'].values,"B_fighter":  df_pred['B_fighter'].values, "R_prob": final_prediction_proba['pred_1'].values,"B_prob": final_prediction_proba['pred_0'].values})

predict_proba.head(10)

#predict_proba.to_csv('predict_probab.csv',index=False)
final_prediction_abs =  pd.DataFrame({"pred":final_xgb_model.predict(X_pred)})



predict_absolute = pd.DataFrame({"R_fighter": df_pred['R_fighter'].values,"B_fighter":  df_pred['B_fighter'].values, "R_prob": final_prediction_abs['pred'].values,"B_prob": 1-final_prediction_abs['pred'].values})

predict_absolute.head(5)

predict_absolute.to_csv('predict_absolute.csv',index=False)