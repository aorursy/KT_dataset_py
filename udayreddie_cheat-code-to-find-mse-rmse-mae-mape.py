# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Fits different models passed in the argument and spits out the metrics

# metrics are calculated on both test and train data - train being in ()

#############################################################################################

def Model_Comparision_Train_Test(AllModels, x_train, y_train, x_test, y_test):

    return_df = pd.DataFrame(columns=['Model', 'MSE', 'RMSE', 'MAE'])

    for myModel in AllModels:

        myModel.fit(x_train, y_train)



        #predict, confusion matrix metrics on train

        y_pred_train = myModel.predict(x_train)

        mse_train, rmse_train, mae_train = extract_metrics_from_predicted(y_train,y_pred_train)

        #print(accuracy_train,sensitivity_train,prec_train,f1score_train)



        #predict, confusion matrix metrics on test

        y_pred_test = myModel.predict(x_test)

        mse_test, rmse_test, mae_test = extract_metrics_from_predicted(y_test, y_pred_test)

        #print(accuracy_test,sensitivity_test,prec_test,f1score_test)



        #create a summary dataframe

        summary = pd.DataFrame([[type(myModel).__name__,

                                         ''.join([str(round(mse_test,3)), "(", str(round(mse_train,3)), ")"]),

                                         ''.join([str(round(rmse_test,3)), "(", str(round(rmse_train,3)), ")"]),

                                         ''.join([str(round(mae_test,3)), "(", str(round(mae_test,3)), ")"])]],

                                         columns=['Model', 'MSE', 'RMSE', 'MAE'])

        return_df = pd.concat([return_df, summary], axis=0)



    #remove index and make model index

    return_df.set_index('Model', inplace=True)

    return(return_df)







def extract_metrics_from_predicted(y_true, y_pred):

    from sklearn.metrics import mean_squared_error,mean_absolute_error 

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_true, y_pred)

    return (mse, rmse,mae)
#base model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



#LR = LinearRegression()

#DTR = DecisionTreeRegressor()

#Abr = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=6), learning_rate=0.01, n_estimators=500)

#Gbr = GradientBoostingRegressor()

Lasso = Lasso()

Ridge = Ridge()

KNNR = KNeighborsRegressor()

RFR = RandomForestRegressor(bootstrap=True,max_depth=80,max_features=3,min_samples_leaf=3, min_samples_split=8, n_estimators=500)

XgbR = XGBRegressor(colsample_bytree=0.9,learning_rate=0.4,n_estimators=500,reg_alpha=0.4)



#skLearn_Model_Comparision_Train_Test([LR, DTR, Abr, Gbr, KNNR, RFR, XgbR], X_train, np.ravel(y_train), X_test, np.ravel(y_test))

skLearn_Model_Comparision_Train_Test([KNNR, RFR, XgbR, Lasso, Ridge], X_train, np.ravel(y_train), X_test, np.ravel(y_test))
def mape(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print('mape:',mape(y_test, preds_test))