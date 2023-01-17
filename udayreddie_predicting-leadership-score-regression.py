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

# metrics: accuracy, precision, sensitivity and F1 score

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
train_data= pd.read_csv('../input/Train-1555063579947.csv')

test_data = pd.read_csv('../input/Test-1555063594850.csv')
train_data.dtypes
test_data.dtypes
test_data.info()
print(train_data.shape)

print(test_data.shape)

train_data.head()

import pandas_profiling as pp



pp.ProfileReport(train_data)
#Dropping the columns which are having more missing values

train_data = train_data.drop(['HadGrievance','Promoted_InLast3Yrs','EmployeeID'],axis=1)
train_data.head()
#SPLITTIG THR DATE COLUMN

x = train_data['ProjectsWorkedOn'].str.split(',', expand=True)

x.columns = ('p1','p2','p3','p4','p5','p6','p7')

y = train_data['DOJ'].str.split('-', expand = True)

y.columns = ('YEAR','MONTH','DAY')

train_data.head()
train_data = pd.concat([train_data,x,y], axis = 1)

train_data.head()
train_data=train_data.drop(['DOJ','ProjectsWorkedOn'],axis=1)

train_data.head()
train_data=train_data.drop(['p4','p5','p6','p7','MONTH','DAY'],axis=1)

train_data.head()
#Correlation

corr = train_data.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
#Function for correlation

corr_matrix = train_data.corr().abs()

# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]

train_data.drop(axis=1,columns=to_drop,inplace=True)
train_data.dtypes
temp = train_data.dtypes[train_data.dtypes == 'object'].index

train_data[temp]=train_data[temp].astype('category')

train_data.dtypes
import seaborn as sns

sns.distplot(train_data['LeadershipPotentialScore'])

print("Skewness: %f" % train_data.LeadershipPotentialScore.skew())

print("Kurtosis: %f" % train_data.LeadershipPotentialScore.kurt())
#MISSING VALUES

missing = train_data.isnull().sum(axis=0).reset_index()

missing.columns = ['column_name', 'missing_count']

missing['missing_ratio'] = (missing['missing_count'] / train_data.shape[0])*100

missing.sort_values(by='missing_ratio', ascending=False)
train_data.dtypes
#Imputing mode for categorical data

train_data['p3'] = train_data['p3'].fillna(train_data['p3'].mode()[0])

train_data['Department'] = train_data['Department'].fillna(train_data['Department'].mode()[0])

train_data['p2'] = train_data['p2'].fillna(train_data['p2'].mode()[0])

# Imputing mean for numerical varibles

train_data['Self_appraisalScore'] = train_data['Self_appraisalScore'].fillna(train_data['Self_appraisalScore'].mean())

train_data['AppraisalRatingScore'] = train_data['AppraisalRatingScore'].fillna(train_data['AppraisalRatingScore'].mean())
train_data.isnull().sum()

print(train_data.shape)
#Dummification

train_data = pd.get_dummies(train_data,columns = ["Department","Earnings_Level","p1","p2","p3","YEAR"],drop_first=True)



print(train_data.shape)

train_data.head()
X= train_data.copy().drop('LeadershipPotentialScore',axis=1)

Y=train_data['LeadershipPotentialScore']
#SPLITTING THE DATA

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
# ## Scale the numeric attributes

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train.iloc[:,:3])



x_train.iloc[:,:3] = scaler.transform(x_train.iloc[:,:3])

x_test.iloc[:,:3] = scaler.transform(x_test.iloc[:,:3])
from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



Lasso = Lasso(alpha=0.01)

Ridge = Ridge(alpha=0.01)

KNNR = KNeighborsRegressor()

RFR = RandomForestRegressor(bootstrap=True,max_depth=80,max_features=3,min_samples_leaf=3, min_samples_split=8, n_estimators=500)

XgbR = XGBRegressor(colsample_bytree=0.9,learning_rate=0.4,n_estimators=500,reg_alpha=0.4)



Model_Comparision_Train_Test([KNNR, RFR, XgbR, Lasso, Ridge], x_train, np.ravel(y_train), x_test, np.ravel(y_test))
#DECESSION TREES

from sklearn.metrics import mean_squared_error

from sklearn.tree import export_graphviz,DecisionTreeRegressor

DTR = DecisionTreeRegressor(random_state=42)

DTR.fit(x_train, y_train)



preds_test_DT = DTR.predict(x_test)

#preds_test_DT = DTR.predict(X_test)



DT_mse_test = mean_squared_error(preds_test_DT, y_test)

DT_rmse_test = np.sqrt(DT_mse_test)



print("Decision Tree Regressor MSE on val: %.4f" %DT_mse_test)

print('Decision Tree Regressor RMSE on val: %.4f' % DT_rmse_test)
test_data = test_data.drop(['HadGrievance','Promoted_InLast3Yrs','EmployeeID'],axis=1)

test_data.head()
#SPLITTIG THR DATE COLUMN

a = test_data['ProjectsWorkedOn'].str.split(',', expand=True)

a.columns = ('p1','p2','p3','p4','p5','p6','p7')

b = test_data['DOJ'].str.split('/', expand = True)

b.columns = ('DAY','MONTH','YEAR')

test_data.head()
test_data = pd.concat([test_data,a,b], axis = 1)

test_data.head()
test_data=test_data.drop(['DOJ','ProjectsWorkedOn'],axis=1)

test_data.head()
test_data=test_data.drop(['p4','p5','p6','p7','MONTH','DAY'],axis=1)

test_data.head()
#MISSING VALUES

missing = test_data.isnull().sum(axis=0).reset_index()

missing.columns = ['column_name', 'missing_count']

missing['missing_ratio'] = (missing['missing_count'] / test_data.shape[0])*100

missing.sort_values(by='missing_ratio', ascending=False)
test_data.HoursWorked_MthlyAvg.nunique
test_data.HoursWorked_MthlyAvg.mode()
#Imputing mode for categorical data

test_data['p3'] = test_data['p3'].fillna(test_data['p3'].mode()[0])

test_data['Department'] = test_data['Department'].fillna(test_data['Department'].mode()[0])

test_data['p2'] = test_data['p2'].fillna(test_data['p2'].mode()[0])

# Imputing mean for numerical varibles

test_data['Self_appraisalScore'] = test_data['Self_appraisalScore'].fillna(test_data['Self_appraisalScore'].mean())

test_data['AppraisalRatingScore'] = test_data['AppraisalRatingScore'].fillna(test_data['AppraisalRatingScore'].mean())

test_data['HoursWorked_MthlyAvg'] = test_data['HoursWorked_MthlyAvg'].fillna(test_data['HoursWorked_MthlyAvg'].mean())
test_data.dropna(how ='all')
test_data.isna().sum(axis=0)
test_data.info()
#Dummification

test_data = pd.get_dummies(test_data,columns = ["Department","Earnings_Level","p1","p2","p3","YEAR"],drop_first=True)



print(test_data.shape)

test_data.head()
test_data.dtypes
test_data.isnull().sum()
#ADABOOSTING

from sklearn.ensemble import AdaBoostRegressor



Ada = AdaBoostRegressor()

Ada.fit(x_train, y_train)



preds_val_Ada = Ada.predict(x_test)

#preds_test_Ada = Ada.predict(X_test)



Ada_mse_val = mean_squared_error(preds_val_Ada, y_test)

Ada_rmse_val = np.sqrt(Ada_mse_val)

print("AdaboostRegressor MSE on val: %.4f" %Ada_mse_val)

print('AdaboostRegressor RMSE on val: %.4f' % Ada_rmse_val)
test_data['HoursWorked_MthlyAvg'] =test_data['HoursWorked_MthlyAvg'].astype('float64')
test_predict =Ada.predict(test_data)
test_data1 = pd.read_csv("../input/Test-1555063594850.csv")



predictions = pd.DataFrame()

predictions['EmployeeID'] = test_data1.EmployeeID

predictions['LeaderShipPotentialScore'] = test_predict
predictions.to_csv("submission_file.csv")