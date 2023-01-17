import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import xgboost as xgb
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
import plotly.express as px
from sklearn.model_selection import GridSearchCV
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
# Reading the train dataset
House_data=pd.read_csv("../input/home-data-for-ml-course/train.csv")
House_data.head()
# Reading the test data set
House_data_test=pd.read_csv("../input/home-data-for-ml-course/test.csv")
House_data_test.head()
 # Merging the test and train datasets, so that all the cleaning can be done at once.
House_data["flag"]="0"
House_data_test["flag"]="1"
final_house_data=pd.concat([House_data,House_data_test])
# Removing these two columns since they have too many null values.
final_house_data.drop(["LotFrontage","GarageYrBlt"],axis=1,inplace=True)

Ordinal_categorical=["MSSubClass","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Condition1"
                    ,"BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","MasVnrType","ExterQual","ExterCond","Foundation"
                    ,"BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir",
                    "Electrical","KitchenQual","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive",
                    "PoolQC","Fence"]
Nominal_categorical=["MSZoning","MiscFeature","SaleType","SaleCondition"]

def nullvalueremovecategoricalcolumns(df_nullcheck,cols):
    for columns in cols:
        if (df_nullcheck[columns].dtypes=='int64')|(df_nullcheck[columns].dtypes=='int32'):
            df_nullcheck[columns].fillna(0,inplace=True)
        df_nullcheck[columns].fillna("unknown",inplace=True)
    print(df_nullcheck[cols].head())
    return df_nullcheck
        
    
Housedata_nullcheck=nullvalueremovecategoricalcolumns(final_house_data,["MSZoning","MiscFeature","SaleType","SaleCondition","MSSubClass","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Condition1"
                    ,"BldgType","Condition2","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation"
                    ,"BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir",
                    "Electrical","KitchenQual","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive",
                    "PoolQC","Fence","Functional","Neighborhood"])
def nullvalueremovenumericcolumns(df_nullcheck_numeric,cols):
    for columns in cols:
        df_nullcheck_numeric[columns].fillna(df_nullcheck_numeric[columns].mean(),inplace=True)
    print(df_nullcheck_numeric[cols].head())
    return df_nullcheck_numeric
Housedata_nullcheck_numeric=nullvalueremovenumericcolumns(final_house_data,["LotArea","YearBuilt","YearRemodAdd",
                                                                     "MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF",
                                                                     "TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF",
                                                                     "GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath",
                                                                     "HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces",
                                                                     "GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch",
                                                                     "3SsnPorch","ScreenPorch","PoolArea","MiscVal","MoSold","YrSold"])
corrs=final_house_data[final_house_data["flag"]=='0'].corr().abs()
s = corrs.unstack()
so = s.sort_values(kind="quicksort",ascending=False)
print(so["SalePrice"])
plt.figure(figsize=(20,10))
sns.barplot(x='ExterQual',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])
plt.figure(figsize=(20,10))
sns.barplot(x='OverallQual',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])
plt.figure(figsize=(20,10))
sns.barplot(x='GarageCars',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])

plt.figure(figsize=(20,10))
sns.barplot(x='BsmtQual',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])

plt.figure(figsize=(20,10))
sns.barplot(x='KitchenQual',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])

plt.figure(figsize=(20,10))
sns.barplot(x='FullBath',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])

plt.figure(figsize=(20,10))
sns.barplot(x='GarageFinish',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])

plt.figure(figsize=(20,10))
sns.barplot(x='TotRmsAbvGrd',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])

from scipy.stats import pearsonr 
corryu,_ =pearsonr(Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"],Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["YearBuilt"])
colorassigned=Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"]
fig = px.scatter(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="YearBuilt", y="SalePrice",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"],Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["GrLivArea"])
colorassigned=Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"]
fig = px.scatter(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="GrLivArea", y="SalePrice",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"],Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["GarageArea"])
colorassigned=Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"]
fig = px.scatter(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="GarageArea", y="SalePrice",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)

from scipy.stats import pearsonr 
corryu,_ =pearsonr(Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"],Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["TotalBsmtSF"])
colorassigned=Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"]
fig = px.scatter(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="TotalBsmtSF", y="SalePrice",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)

from scipy.stats import pearsonr 
corryu,_ =pearsonr(Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"],Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["1stFlrSF"])
colorassigned=Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"]
fig = px.scatter(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="1stFlrSF", y="SalePrice",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)


from scipy.stats import pearsonr 
corryu,_ =pearsonr(Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"],Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["YearRemodAdd"])
colorassigned=Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"]
fig = px.scatter(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="YearRemodAdd", y="SalePrice",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)

from scipy.stats import pearsonr 
corryu,_ =pearsonr(Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"],Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["MasVnrArea"])
colorassigned=Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"]
fig = px.scatter(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="MasVnrArea", y="SalePrice",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)
plt.figure(figsize=(20,10))
sns.barplot(x='Fireplaces',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])

plt.figure(figsize=(20,10))
sns.barplot(x='FireplaceQu',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])

plt.figure(figsize=(20,10))
sns.barplot(x='GarageType',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])

plt.figure(figsize=(20,10))
sns.barplot(x='HeatingQC',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"],Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["BsmtFinSF1"])
colorassigned=Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"]
fig = px.scatter(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="BsmtFinSF1", y="SalePrice",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)

plt.figure(figsize=(20,10))
sns.barplot(x='Foundation',y='SalePrice',data=Housedata_nullcheck[Housedata_nullcheck['flag']=='0'])
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"],Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["WoodDeckSF"])
colorassigned=Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"]
fig = px.scatter(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="WoodDeckSF", y="SalePrice",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"],Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["2ndFlrSF"])
colorassigned=Housedata_nullcheck[Housedata_nullcheck['flag']=='0']["SalePrice"]
fig = px.scatter(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="2ndFlrSF", y="SalePrice",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)

fig = px.pie(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], values='HalfBath', names='HalfBath')
fig.show()
fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="ExterQual", 
             y="SalePrice", points="all",color="ExterQual",
             title="Distribution of SalePrice with External Quality of House",
            )
fig.show()
fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="GarageCars", 
             y="SalePrice", points="all",color="GarageCars",
             title="Distribution of SalePrice with number of Car Garages in House",
            )
fig.show()
fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="OverallQual", 
             y="SalePrice", points="all",color="OverallQual",
             title="Distribution of SalePrice with Overall present Quality of House",
            )
fig.show()
fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="BsmtQual", 
             y="SalePrice", points="all",color="BsmtQual",
             title="Distribution of SalePrice with the quality of the basement in the House",
            )
fig.show()

fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="KitchenQual", 
             y="SalePrice", points="all",color="KitchenQual",
             title="Distribution of SalePrice with the quality of the Kitchen in the House",
            )
fig.show()
fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="FullBath", 
             y="SalePrice", points="all",color="FullBath",
             title="Distribution of SalePrice with the number of full bathrooms in the House",
            )
fig.show()

fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="GarageFinish", 
             y="SalePrice", points="all",color="GarageFinish",
             title="Distribution of SalePrice with the status of Garage in the House",
            )
fig.show()

fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="TotRmsAbvGrd", 
             y="SalePrice", points="all",color="TotRmsAbvGrd",
             title="Distribution of SalePrice with the total number of rooms above the ground in the House",
            )
fig.show()

fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="Fireplaces", 
             y="SalePrice", points="all",color="Fireplaces",
             title="Distribution of SalePrice with the number of fireplaces present in the House",
            )
fig.show()
      
     
fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="FireplaceQu", 
             y="SalePrice", points="all",color="FireplaceQu",
             title="Distribution of SalePrice with the quality of fireplaces present in the House",
            )
fig.show()

fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="GarageType", 
             y="SalePrice", points="all",color="GarageType",
             title="Distribution of SalePrice with the type of garage present in the House",
            )
fig.show()

fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="HeatingQC", 
             y="SalePrice", points="all",color="HeatingQC",
             title="Distribution of SalePrice with the quality of Heating in the House",
            )
fig.show()

fig = px.box(Housedata_nullcheck[Housedata_nullcheck['flag']=='0'], x="Foundation", 
             y="SalePrice", points="all",color="Foundation",
             title="Distribution of SalePrice with the type of material used for constructing the House",
            )
fig.show()

def labelencoding(df,cols):
    for columns in cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[columns] = le.fit_transform(df[columns].values)
    print(df[cols].head())
    return df
    
Housedata_encoded=labelencoding(final_house_data,["MSZoning","MiscFeature","SaleType","SaleCondition","MSSubClass","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Condition1"
                    ,"BldgType","Condition2","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation"
                    ,"BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir",
                    "Electrical","KitchenQual","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive",
                    "PoolQC","Fence","Functional","Neighborhood"])
# Checking the null values we do not have any null values after the cleaning, only the saleprice has since it is the target variable.
ax=plt.figure(figsize=(20,10))
sns.heatmap(Housedata_nullcheck.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Finding the correlation of all the columns with the target variable, after encoding the categorical columns.
corrs=final_house_data[final_house_data["flag"]=='0'].corr().abs()
s = corrs.unstack()
so = s.sort_values(kind="quicksort",ascending=False)
print(so["SalePrice"])
# Plotting the count of few columns
colorassigned=Housedata_encoded["OverallQual"]
fig = px.histogram(final_house_data, x="OverallQual", marginal="rug",
                   hover_data=final_house_data.columns,nbins=30,color=colorassigned)
fig.show()
colorassigned=Housedata_encoded["GarageCars"]
fig = px.histogram(final_house_data, x="GarageCars", marginal="rug",
                   hover_data=final_house_data.columns,nbins=20,color=colorassigned)
fig.show()
colorassigned=Housedata_encoded["ExterQual"]
fig = px.histogram(final_house_data, x="ExterQual", marginal="rug",
                   hover_data=final_house_data.columns,nbins=30,color=colorassigned)
fig.show()

final_house_data.kurtosis(axis=0) 
# Scaling the columns that have too much Kurtosis.
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
final_house_data[['3SsnPorch','Condition2', 'PoolArea','Utilities','MiscVal','Heating','LotArea','LowQualFinSF','Street','RoofMatl','MiscFeature','EnclosedPorch']] = mms.fit_transform(final_house_data[['3SsnPorch','Condition2', 'PoolArea','Utilities','MiscVal','Heating','LotArea','LowQualFinSF','Street','RoofMatl','MiscFeature','EnclosedPorch']])
# Selecting the train part of the dataset by making flag=0
p=final_house_data[final_house_data["flag"]=='0']

# Removing the flag, saleprice and id columns from the features matrix
colss = [col for col in p.columns if col not in ['flag','SalePrice','Id']]
X=p[colss]
X.head()
y=p["SalePrice"]
# importing train test set
from sklearn.model_selection import train_test_split
# splitting the training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# importing linear regressor
from sklearn.linear_model import LinearRegression
# Instantiating linear regressor
lm=LinearRegression()
# Fitting the model on the training data
lm.fit(X_train,y_train)
# Prediction on the training data
predictions_linearregressor_traindata=lm.predict(X_test)
dff = pd.DataFrame({'Actual': y_test, 'Predicted': predictions_linearregressor_traindata})
dff
# Checking the mean squared log error on training data
from sklearn import metrics
metrics.mean_squared_log_error(y_test, predictions_linearregressor_traindata)
# Checking the score of the Linear regressor on training data
linearregressionscore=lm.score(X_test,y_test)
linearregressionscore
# Selecting the test part of the data
Lineartestdata=final_house_data[final_house_data["flag"]=='1']
Lineartestdata.head()
# Removing the 3 unwanted features that will impact the prediction, and SalePrice is target variable so cannot be in feature matrix.
testdata = Lineartestdata.drop(['flag','SalePrice','Id'], axis=1)
testdata.head()
# making predictions on the test data set
predictions_linearregressor_testdata=lm.predict(testdata)
predictions_linearregressor_testdata

# Importing the required libraries
from sklearn.model_selection import train_test_split
# Splitting the train and the test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Importing the deicion tree regressor
from sklearn.tree import DecisionTreeRegressor
# Instantiating the DecisionTree regressor
decisiontreereg=DecisionTreeRegressor()
# Fitting on the training data set
decisiontreereg.fit(X_train,y_train)
# Predictions for the training data sets
predictions_decisiontree_traindata=decisiontreereg.predict(X_test)
predictions_decisiontree_traindata
# Score for the decisiontree regressor on the training data set
decisiontreescore=decisiontreereg.score(X_test,y_test)
decisiontreescore
# Getting the decisiontree test data set using the flag filtering
Decisiontreetestdata=final_house_data[final_house_data["flag"]=='1']
# Removing the unwanted columns
testdata_decisiontree = Decisiontreetestdata.drop(['flag','SalePrice','Id'], axis=1)
# Making predictions on the test data set
predictions_decisiontree_testdata=decisiontreereg.predict(testdata_decisiontree)
predictions_decisiontree_testdata
# Getting the train data using the flag variable.
q=final_house_data[final_house_data["flag"]=='0']
# Filtering the columns 
cold = [col for col in q.columns if col not in ['flag','SalePrice','Id']]
Z=q[cold]
t=q['SalePrice']
# Importing the required 
from sklearn.model_selection import train_test_split
# Splitting the training and the test set within the train dat set
Z_train, Z_test, t_train, t_test = train_test_split(Z, t, test_size=0.3, random_state=42)
# Importing the randomforest regressor
from sklearn.ensemble import RandomForestRegressor
# Instantiating the regressor and passing the required parameters to the regressor
Randomforestregr=RandomForestRegressor(n_estimators = 100,n_jobs = -1,oob_score = True, bootstrap = True,random_state=42)
# Fitting to the training data set
Randomforestregr.fit(Z_train,t_train)
# Pedicting on the trained dataset
prediction_randomforest_traindata=Randomforestregr.predict(Z_test)
# Score on the train dataset
randomforestscore=Randomforestregr.score(Z_test,t_test)
randomforestscore
print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(Randomforestregr.score(Z_train, t_train), 
                                                                                             Randomforestregr.oob_score_,
                                                                                             Randomforestregr.score(Z_test, t_test)))
# Getting the test dataset to make the final predictions
Randomforesttestdata=final_house_data[final_house_data["flag"]=='1']
# removing the unwanted columns
testdata_randomforest = Randomforesttestdata.drop(['flag','SalePrice','Id'], axis=1)
# Making the predictions on the test dataset.
predictions_randomforest_testdata=Randomforestregr.predict(testdata_randomforest)
# Getting the importances of the features of the dataset. The significance of each feature is shown in predicting the SalePrice.
feature_imp=pd.DataFrame(sorted(zip(Randomforestregr.feature_importances_,Z)),columns=["Significance","Features"])
fig=plt.figure(figsize=(20,20))
sns.barplot(x="Significance",y="Features",data=feature_imp.sort_values(by="Significance",ascending=False),dodge=False)
plt.title("Important features for predicting the SalePrice of the House")
plt.tight_layout()
plt.show()
# Getting the train data from the dataset 
r=final_house_data[final_house_data["flag"]=='0']
# Getting the required columns
colsxg=[col for col in q.columns if col not in ['flag','SalePrice','Id']]
A=r[colsxg]
b=r['SalePrice']
# Splitting the train and test set within the train set
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.3, random_state=42)
# Instantiated the XgBoost model by passing the most optimal parameters that were obtained by performing the GridSearchcv.
XgBoostmodel = xgb.XGBRegressor(
    n_estimators=100,
    reg_lambda=1,
    reg_alpha=0.002,
    gamma=0.3,
    max_depth=4,
    min_child_weight=4,
    subsample=1,
    colsample_bytree=1,
)
# Fitting on the train sample.
XgBoostmodel.fit(A_train,b_train)
# Making predictions on the train sample
Xgboost_prediction_train = XgBoostmodel.predict(A_test)
Xgboost_prediction_train
# Score on training set
Xgboostscore=XgBoostmodel.score(A_test,b_test)
Xgboostscore
# Calculated the mean squared log error
metrics.mean_squared_log_error(b_test, Xgboost_prediction_train)
print('\n Best hyperparameters:')
print(random_search.best_params_)
# Getting the test dataset from the whole dataset
Xgboosttestdata=final_house_data[final_house_data["flag"]=='1']

testdata_xgboost=Randomforesttestdata.drop(['flag','SalePrice','Id'], axis=1)
# Making predictions on the test data set
predictions_xgboost_testdata=XgBoostmodel.predict(testdata_xgboost)
# Exporting the results to a dataframe from an array and then converting it to csv file for export
resultcsv=pd.DataFrame(predictions_xgboost_testdata)
resultcsv.shape
resultcsv.to_csv('Result.csv')
# Creating a dictionary for all the models to store thier scores and convert this to dataframe.
dict={"Linear Regressor":[linearregressionscore],"DecisionTree Regressor":[decisiontreescore],"RandomForest Regressor":[randomforestscore],"XGBoost Regressor":[Xgboostscore]}
df_comparison_models=pd.DataFrame(dict,["Score"])
# Plotting the performance of all the 3 models on the train dataset.
%matplotlib inline
model_accuracy = pd.Series(data=[linearregressionscore,decisiontreescore,randomforestscore,Xgboostscore], 
        index=['Linear Regressor','DecisionTree Regressor','RandomForest Regressor','XGBoost Regressor'])
fig= plt.figure(figsize=(8,8))
model_accuracy.sort_values().plot.barh()
plt.title('Model Accuracy')