# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np

import sklearn

%matplotlib inline 



from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import RandomForestRegressor , BaggingRegressor , AdaBoostRegressor , GradientBoostingRegressor , HistGradientBoostingRegressor , VotingRegressor , StackingRegressor 

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor



import warnings

warnings.filterwarnings('ignore')

from plotly.offline import init_notebook_mode, iplot

from plotly import graph_objs as go

init_notebook_mode(connected=True)



from fbprophet import Prophet
def final_func(df,xcol,ycol,estimator,est_desc = "default",train_ratio = 0.7 , val_ratio = 0.2 , test_ratio = 0.1, data_given = False , piece_wise = True , show_test = False ) : 

  #assert(train_ratio+test_ratio+val_ratio == 1.0)

    if piece_wise == True :

        if data_given == False :

            data = []

            for i in range(7) :

                data.append(df.loc[df["a"+str(i)] == 1,xcol+ycol].reset_index())

            print(data[0].shape)

        else :

            data = df 

        y_pred_train = []

        y_pred_val = []

        y_pred_test = []

        y_train_ = []

        y_val_ = []

        y_test_ = []



        est = []

        for i in range(3,4) :

            plt.plot(data[i].label,"b")

            plt.plot(data[i].y_hat,"g")

            train_ind = int(data[i].shape[0]*train_ratio)

            val_ind = int(data[i].shape[0]*train_ratio) + int(data[i].shape[0]*val_ratio)

            x_train , x_val , x_test , y_train , y_val , y_test = data[i].loc[:train_ind,xcol],data[i].loc[train_ind+1:val_ind,xcol],data[i].loc[val_ind+1:,xcol],data[i].loc[:train_ind,ycol],data[i].loc[train_ind+1:val_ind,ycol],data[i].loc[val_ind+1:,ycol]

            print(x_train.shape , x_val.shape , x_test.shape , y_train.shape , y_val.shape , y_test.shape)

            estimator.fit(x_train,y_train)

            est.append(estimator)

            y_pred_train_ = estimator.predict(x_train) 

            y_pred_val_ = estimator.predict(x_val)

            if show_test :

                y_pred_test_ = estimator.predict(x_test)

            if show_test :

                y_test_ += list(y_test.values)     

            y_train_ += list(y_train.values)

            y_val_ += list(y_val.values)

            y_pred_train += list(y_pred_train_)

            y_pred_val += list(y_pred_val_)

            if show_test :

                y_pred_test += list(y_pred_test_)



            joblib.dump(estimator, 'mode_p_'+est_desc+'_'+str(i)+'.pkl')



        print("val rmse %.4f"%((np.sqrt(mean_squared_error(y_pred = y_pred_val,y_true = y_val_)))))

        print("train rmse %.4f"%((np.sqrt(mean_squared_error(y_pred = y_pred_train,y_true = y_train_)))))

        if show_test :

            print("test %.4f"%((np.sqrt(mean_squared_error(y_pred = y_pred_test,y_true = y_test_)))))

        return estimator

    else :

        train_ind = int(df.shape[0]*train_ratio)

        val_ind = int(df.shape[0]*train_ratio) + int(df.shape[0]*val_ratio)

        x_train , x_val , x_test , y_train , y_val , y_test = df.loc[:train_ind,xcol],df.loc[train_ind+1:val_ind,xcol],df.loc[val_ind+1:,xcol],df.loc[:train_ind,ycol],df.loc[train_ind+1:val_ind,ycol],df.loc[val_ind+1:,ycol]

        print(x_train.shape , x_val.shape , x_test.shape , y_train.shape , y_val.shape , y_test.shape)

        estimator.fit(x_train,y_train)

        y_pred_train = estimator.predict(x_train) 

        y_pred_val = estimator.predict(x_val)

        if show_test :

            y_pred_test = estimator.predict(x_test)

        print("val rmse %.4f"%(np.sqrt(mean_squared_error(y_pred = y_pred_val,y_true = y_val))))

        print("train rmse %.4f"%(np.sqrt(mean_squared_error(y_pred = y_pred_train,y_true = y_train))))

        if show_test :

            print("test %.4f"%(np.sqrt(mean_squared_error(y_pred = y_pred_test,y_true = y_test))))

        joblib.dump(estimator, 'mode_f_'+est_desc+'.pkl')

        return estimator



df = pd.read_csv("/kaggle/input/bits-f464-l1/train.csv")

df.drop(columns = ["id"],inplace = True)

# est = [GridSearchCV((n_estimators = 200),param_grid={"base_estimator":[DecisionTreeRegressor()]},n_jobs = -1)]# est = [GridSearchCV(AdaBoostRegressor(base_estimator = DecisionTreeRegressor(),n_estimators=100,),param_grid={"loss":["linear"]},n_jobs= -1)]#,HistGradientBoostingRegressor(l2_regularization = 100)]

est = [GridSearchCV(AdaBoostRegressor(base_estimator = DecisionTreeRegressor(),n_estimators=150,),param_grid={"loss":["linear"]},n_jobs= -1)]#,HistGradientBoostingRegressor(l2_regularization = 100)]

for e in est :

    print(e)

    est = final_func(df,[x for x in df.columns if x!="label"],["label"],e,train_ratio = 0.8 , val_ratio = 0.2 , test_ratio = 0.0 ,data_given = True , piece_wise = False , show_test = False )

df_ss =  pd.read_csv("/kaggle/input/bits-f464-l1/sampleSubmission.csv")

df_test = pd.read_csv("/kaggle/input/bits-f464-l1/test.csv")

df_test.drop(columns = ["id"],inplace = True) 
df_ss["label"] = est.predict(df_test)
df_ss.to_csv("2017A7PS0943G.csv",index = False)