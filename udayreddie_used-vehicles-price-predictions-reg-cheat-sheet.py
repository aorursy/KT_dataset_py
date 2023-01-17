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

def skLearn_Model_Comparision_Train_Test(myModels, x_train, y_train, x_test, y_test):

    return_df = pd.DataFrame(columns=['Model', 'MSE', 'RMSE', 'MAE'])

    for myModel in myModels:

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

        current_summary = pd.DataFrame([[type(myModel).__name__,

                                         ''.join([str(round(mse_test,3)), "(", str(round(mse_train,3)), ")"]),

                                         ''.join([str(round(rmse_test,3)), "(", str(round(rmse_train,3)), ")"]),

                                         ''.join([str(round(mae_test,3)), "(", str(round(mae_test,3)), ")"])]],

                                         columns=['Model', 'MSE', 'RMSE', 'MAE'])

        return_df = pd.concat([return_df, current_summary], axis=0)



    #remove index and make model index

    return_df.set_index('Model', inplace=True)

    return(return_df)



#############################################################################################

# Extract classification metrics from predicted and actual values

#############################################################################################

def extract_metrics_from_predicted(y_true, y_pred):

    from sklearn.metrics import mean_squared_error,mean_absolute_error 

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_true, y_pred)

    return (mse, rmse,mae)
vData = pd.read_csv("../input/TrainData.csv")
# split X and y

vData_y = vData[['Price']]

vData_X = vData.drop(['Price'], axis=1)

vData_X.head()

vData_X.info()

vData_X.isna().sum()

vData_X.shape
vData.head()
vData_X.OfferType.value_counts() # single value can be deleted

vData_X.SellerType.value_counts() # single value can be deleted

vData_X.NumberOfPictures.value_counts()

vData_X.drop(['OfferType', 'SellerType', 'NumberOfPictures', 'DataCollectedDate', 'BrandOfTheVehicle', 'ZipCode'], axis=1, inplace=True)
#Date related feilds manipulation



from datetime import datetime



#for dt in vData.DataCollectedDate

vData_X.DateOfAdCreation = pd.to_datetime(vData_X.DateOfAdCreation, dayfirst=True)

vData_X.DateOfAdLastSeen = pd.to_datetime(vData_X.DateOfAdLastSeen, dayfirst=True)

vData_X['DaysDiff'] = (vData_X['DateOfAdLastSeen'] - vData_X['DateOfAdCreation']).dt.days
#temp day - assumed to be 1 for all observations

vData_X['RegDay'] = 1

vData_X['RegDate'] = vData_X[['YearOfVehicleRegistration', 'MonthOfVehicleRegistration', 'RegDay']].apply(lambda s : datetime(*s),axis = 1)

vData_X['VehicleAge'] = (vData_X['DateOfAdCreation'] - vData_X['RegDate']).dt.days
vData_X.drop(['DateOfAdLastSeen', 'DateOfAdCreation', 'RegDay'], axis=1, inplace=True)

vData_X.drop(['YearOfVehicleRegistration', 'MonthOfVehicleRegistration'], axis=1, inplace=True)

vData_X.drop(['RegDate'], axis=1, inplace=True)

vData_X.drop(['NameOfTheVehicle', 'VehicleID'], axis=1, inplace=True)

vData_X.drop(['ModelOfTheVehicle'], axis=1, inplace=True)

vData_X.info()
#convert the object types to categorical

vData_X[vData_X.select_dtypes(['object']).columns] = vData_X.select_dtypes(['object']).apply(lambda x: x.astype('category'))
vData_X.info()
vData_X.GearBoxType.unique()
def categorical_count_plot(data):

    import seaborn as sns

    import pandas as pd

    import matplotlib.pyplot as plt

    sns.set(style="whitegrid", rc={'figure.figsize':(8,8)})



    cColumns = data.select_dtypes(['category']).columns

    print(cColumns)

    fig, axes =plt.subplots(3,2,figsize=(10,10))



    sns.countplot(vData['VehicleType'], ax=axes[0,0])

    sns.countplot(vData['GearBoxType'], ax=axes[0,1])

    sns.countplot(vData['TypeOfTheFuelUsed'], ax=axes[1,0], orient='h')

    sns.countplot(vData['BrandOfTheVehicle'], ax=axes[1,1])

    sns.countplot(vData['IsDamageRepaired'], ax=axes[2,0])

    plt.xticks(rotation=90)

    plt.show()



categorical_count_plot(vData)
vData_X.isna().sum()
def plot_na_percentage(data_X, thresholdPercent):

    #first plot the output

    import seaborn as sns

    df = pd.DataFrame(data_X.isnull().sum()/len(data_X)*100, columns=['NaNPercent'])

    df_sort = df.sort_values(by='NaNPercent',ascending=False)

    df_sort = df_sort[df_sort > (thresholdPercent*100)].dropna()

    sns.barplot(x=df_sort.index, y=df_sort['NaNPercent'])
plot_na_percentage(vData_X, 0.0001) 



# from graph four columns has missing values and are in permissible range

# all the missing columns seems to be categorical variable

# replace them with mode
#handle missing values



print(vData_X.isna().sum().sum())

vData_X['VehicleType'] = vData_X['VehicleType'].fillna("limousine")

vData_X['GearBoxType'] = vData_X['GearBoxType'].fillna("manual")

vData_X['IsDamageRepaired'] = vData_X['IsDamageRepaired'].fillna("No")

vData_X['TypeOfTheFuelUsed'] = vData_X['TypeOfTheFuelUsed'].fillna("petrol")

print(vData_X.isna().sum().sum())
vData_X.columns
vData_X = pd.get_dummies(vData_X)

vData_X.head()
def rs_fs_VIF(data_X, thresh=5):

    from statsmodels.stats.outliers_influence import variance_inflation_factor 

    import pandas as pd

    

    cols = data_X.columns

    variables = np.arange(data_X.shape[1])

    dropped=True

    while dropped:

        df = pd.DataFrame(index=cols[variables])

        dropped=False

        c = data_X[cols[variables]].values

        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

        df['VIF'] = vif

        print(df)

        maxloc = vif.index(max(vif))

        #print(vif)

        if max(vif) > thresh:

            print('dropping \'' + data_X[cols[variables]].columns[maxloc])

            variables = np.delete(variables, maxloc)

            dropped=True



    print('Remaining variables:')

    print(data_X.columns[variables])

    return data_X[cols[variables]]
def remove_highly_correlated(data_X, corThreshold, plot=False):

    

    #generate the plot of required

    if plot:

        plot_correlation_heatmap(data_X)



    # Create correlation matrix

    corr_matrix = data_X.corr().abs()    

    # Select upper triangle of correlation matrix

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))    

    # Find index of feature columns with correlation greater than threshold

    to_drop = [column for column in upper.columns if any(upper[column] > corThreshold)]    

    #drop the columns as formulated above

    print("No of columns dropped for high correlation :", len(to_drop))

    return(data_X.drop(columns=to_drop))
#This method provides a sns heatmap on the data_X

def plot_correlation_heatmap(data_X):

    

    import seaborn as sns

    import matplotlib.pyplot as plt

    #Generate a mask for the upper triangle

    corr = data_X.corr()

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True  

    #Set up the matplotlib figure

    f, ax = plt.subplots(figsize=(12, 10))    

    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plot_correlation_heatmap(vData_X)
remove_highly_correlated(vData_X, corThreshold=0.9, plot=True)
#train test split to test model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(vData_X, vData_y, test_size=0.2, random_state=1107)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#base model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge



Lasso = Lasso(alpha=0.01)

Ridge = Ridge(alpha=0.01)

skLearn_Model_Comparision_Train_Test([Lasso, Ridge], X_train, np.ravel(y_train), X_test, np.ravel(y_test))

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
def rs_fs_randomForestSelection(X_train, y_train, n=-1):

    

    from sklearn.ensemble import RandomForestRegressor

    import matplotlib.pyplot as plt

    %matplotlib inline

    import seaborn as sns

    sns.set(style="whitegrid", rc={'figure.figsize':(7,5)})

    

    rfc = RandomForestRegressor()

    rfc.fit(X_train, y_train)



    fi = pd.DataFrame()

    fi['Features'] = X_train.columns



    fi['Importance%'] = rfc.feature_importances_*100

    fi_sorted = fi.sort_values(['Importance%', 'Features'], ascending = [False, True])

    

    if n != -1:

        fi_sorted = fi_sorted.head(n)

    

    print(fi_sorted)

    sns.barplot(fi_sorted['Features'], fi_sorted['Importance%'])

    plt.xticks(rotation=90)
X_train_new = X_train[['VehicleAge', 'PowerOfTheEngine', 'DistranceTravelled', 'DaysDiff']]

X_test_new = X_test[['VehicleAge', 'PowerOfTheEngine', 'DistranceTravelled', 'DaysDiff']]



#base model

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor



RFR = RandomForestRegressor(bootstrap=True,max_depth=80,max_features=3,min_samples_leaf=3, min_samples_split=8, n_estimators=500)

XgbR = XGBRegressor(colsample_bytree=0.9,learning_rate=0.4,n_estimators=500,reg_alpha=0.4)



skLearn_Model_Comparision_Train_Test([RFR, XgbR], X_train_new, np.ravel(y_train), X_test_new, np.ravel(y_test))
rs_fs_randomForestSelection(X_train, y_train)
from datetime import datetime



test_data = pd.read_csv("../input/TestData.csv")

test_data.isna().sum()

print(test_data.isna().sum().sum())
#vData_X.drop(['OfferType', 'SellerType', 'NumberOfPictures', 'DataCollectedDate', 'BrandOfTheVehicle', 'ZipCode'], axis=1, inplace=True) # 4columns removed

test_data.drop(['NumberOfPictures', 'DataCollectedDate', 'ZipCode'], axis=1, inplace=True) # 4columns removed



#for dt in vData.DataCollectedDate

test_data.DateOfAdCreation = pd.to_datetime(test_data.DateOfAdCreation, dayfirst=True)

test_data.DateOfAdLastSeen = pd.to_datetime(test_data.DateOfAdLastSeen, dayfirst=True)

test_data['DaysDiff'] = (test_data['DateOfAdLastSeen'] - test_data['DateOfAdCreation']).dt.days



#temp day - assumed to be 1 for all observations

test_data['RegDay'] = 1

test_data['RegDate'] = test_data[['YearOfVehicleRegistration', 'MonthOfVehicleRegistration', 'RegDay']].apply(lambda s : datetime(*s),axis = 1)

test_data['VehicleAge'] = (test_data['DateOfAdCreation'] - test_data['RegDate']).dt.days



test_data.drop(['DateOfAdLastSeen', 'DateOfAdCreation', 'RegDay'], axis=1, inplace=True)

test_data.drop(['YearOfVehicleRegistration', 'MonthOfVehicleRegistration'], axis=1, inplace=True)

test_data.drop(['RegDate'], axis=1, inplace=True)

test_data.drop(['NameOfTheVehicle', 'VehicleID'], axis=1, inplace=True)

test_data.drop(['ModelOfTheVehicle'], axis=1, inplace=True)

test_data.info()



test_data[test_data.select_dtypes(['object']).columns] = test_data.select_dtypes(['object']).apply(lambda x: x.astype('category'))



print(test_data.isna().sum().sum())

test_data['VehicleType'] = test_data['VehicleType'].fillna("limousine")

test_data['GearBoxType'] = test_data['GearBoxType'].fillna("manual")

test_data['IsDamageRepaired'] = test_data['IsDamageRepaired'].fillna("No")

test_data['TypeOfTheFuelUsed'] = test_data['TypeOfTheFuelUsed'].fillna("petrol")

print(test_data.isna().sum().sum())

missing_cols = set(X_train.columns ) - set(test_data.columns )

# Add a missing column in test set with default value equal to 0

for c in missing_cols:

    test_data[c] = 0



# Ensure the order of column in the test set is in the same order than in train set

test1 = test_data[X_train.columns]
missing_cols = set(X_train.columns ) - set(test1.columns )

missing_cols
print(len(X_train.columns))

print(len(test_data.columns))
from xgboost import XGBRegressor

XgbR = XGBRegressor(colsample_bytree=0.9,learning_rate=0.4,n_estimators=500,reg_alpha=0.4)

XgbR.fit(X_train, y_train)
test_predict = XgbR.predict(test1)
test_predict
test_data1 = pd.read_csv("../input/TestData.csv")



predictions = pd.DataFrame()

predictions['VehicleID'] = test_data1.VehicleID

predictions['Price'] = test_predict
predictions.to_csv("Submission_file")