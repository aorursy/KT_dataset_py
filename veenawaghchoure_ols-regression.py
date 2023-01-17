import pandas as pd
import numpy as np
from scipy.stats import shapiro
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import cross_val_score
#Importing the data DC_Properties to a pandas dataframe
dc_prop=pd.read_csv('../input/DC_Properties.csv', low_memory = False)
# dc_prop.head()
#Gives the number of null values in each column
null_columns=dc_prop.columns[dc_prop.isnull().any()]
dc_prop[null_columns].isnull().sum()
#Getting the unknown data
unknown_data=dc_prop[dc_prop["PRICE"].isnull()]
# unknown_data.head()
#Removing the unknown data from the original data
final_dc_prop=pd.concat([dc_prop, unknown_data, unknown_data]).drop_duplicates(keep=False)
# final_dc_prop.head()
null_columns=final_dc_prop.columns[final_dc_prop.isnull().any()]
final_dc_prop[null_columns].isnull().sum()

#So there are no null values of price in the final_dc_prop data
sns.distplot(final_dc_prop["PRICE"])
plt.show()
sns.boxplot(final_dc_prop["PRICE"])
plt.show()
final_dc_prop["PRICE1"] = final_dc_prop["PRICE"]**(1./3)
final_dc_prop = final_dc_prop[np.abs(final_dc_prop["PRICE1"]-final_dc_prop["PRICE1"].mean()) <= (3*final_dc_prop["PRICE1"].std())]
final_dc_prop.drop(columns = "PRICE", inplace = True)
sns.distplot(final_dc_prop["PRICE1"])
plt.show()
### Missing values in the data
null_columns = final_dc_prop.columns[final_dc_prop.isnull().any()]
n = len(final_dc_prop.index)
null = final_dc_prop[null_columns].isnull().sum()*100/n
# null
rem = []
for i in range(0,len(null)):
    if null[i] > 30:
        rem.append(null.index[i])
# rem
     
#Dropping columns with missing values more than 30%
final_dc_prop.drop(columns = rem, inplace = True)
final_dc_prop['AC']=final_dc_prop['AC'].astype('category')
final_dc_prop['AC'].replace('0',np.nan, inplace = True)
final_dc_prop['AYB'].describe()
final_dc_prop['QUADRANT'].value_counts()
final_dc_prop['AYB'].fillna((final_dc_prop['AYB'].mean()), inplace=True)
final_dc_prop['QUADRANT'].fillna("NW", inplace=True)
final_dc_prop["HEAT"].value_counts()
final_dc_prop["HEAT"] = final_dc_prop["HEAT"].replace(to_replace = "No Data", value = np.nan)

final_dc_prop["ZIPCODE"] = final_dc_prop["ZIPCODE"].astype('category')
#CREATING DUMMY VARIABLES FOR CATEGORICAL DATA IN  DATA
dummy = pd.get_dummies(final_dc_prop.loc[:,["HEAT",'AC',"QUALIFIED","SOURCE","ASSESSMENT_NBHD","ASSESSMENT_SUBNBHD","WARD","QUADRANT", "ZIPCODE"]])

#Dropping Categories which we have replaced by dummy variables in  data
final_dc_prop.drop(['Unnamed: 0','HEAT','QUALIFIED','SOURCE',"ASSESSMENT_SUBNBHD",'ASSESSMENT_NBHD','WARD','QUADRANT','AC','X','Y',"LATITUDE","LONGITUDE",'SQUARE','ZIPCODE'],axis=1,inplace=True)
# final_dc_prop.drop(['Unnamed: 0','HEAT','QUALIFIED','SOURCE','ASSESSMENT_NBHD','WARD','QUADRANT','AC'],axis=1,inplace=True)

final_dc_prop = final_dc_prop.join(dummy)
#Removing all NA values from data
final_dc_prop.dropna(inplace=True)
#Extracting year from Saledate and storing it as a column sale_date in thedataset
date=final_dc_prop['SALEDATE']
date=date.tolist()
type(date)
date[1]
new_date=[]
for i in range(0,len(date)):
    g=str(date[i]).split('-')
    new_date.append(g[0])
new_date
new_date = list(map(int, new_date))
new_date
sale_date = list(map(lambda x: 2018-x, new_date))
# sale_date
se=pd.Series(sale_date)
final_dc_prop['saledate']=se.values
# train_new.head()
#AYB is the earliest time the main portion of the building was build, so we subtract it from the 
#year of the saledate in order to know that after how many years the building was sold after its
#main portion was build
ayb=final_dc_prop['AYB']
ayb=ayb.tolist()
ayb_saledate=np.subtract(new_date,ayb)
ayb_saledate=ayb_saledate.tolist()

#EYB is the year an improvement was built more recent than actual year built, so we subtract it from the 
#year of the saledate in order to know that after how many years the building was sold after its
#first improvement was built.
eyb=final_dc_prop['EYB']
eyb=eyb.tolist()
eyb_saledate=np.subtract(new_date,eyb)
eyb_saledate=eyb_saledate.tolist()
#eyb_saledate
#Adding ayb_saledate and eyb_saledate which consists of the number of years after which the building 
#was sold after its main portion was build and the number of years after which the building was sold
#after its first improvement was built respectively
se=pd.Series(ayb_saledate)
final_dc_prop['ayb_saledate']=se.values
#train_new.head()
se=pd.Series(eyb_saledate)
final_dc_prop['eyb_saledate']=se.values
# train_new.head()
#dropping some redundant columns from dataset
final_dc_prop.drop(['GIS_LAST_MOD_DTTM','SALEDATE','AYB','EYB'],axis=1,inplace=True)
#Dividing the data final_dc_prop into trainining and test
msk = np.random.rand(len(final_dc_prop)) < 0.8
train_new = final_dc_prop[msk]
test_new = final_dc_prop[~msk]
#Separating the data into response and regressors.
X=train_new.drop(['PRICE1'],axis=1)
Y=train_new['PRICE1']
X=X.reset_index();
#X.head()
Y=Y.reset_index();
#Y.head()
Y=Y.drop('index',axis=1);
#Y.head()
X=X.drop('index',axis=1);
#list(X.columns.values)
train_new.head()
#Fitting the regular linear regression model to the training data set
import statsmodels.api as sm

X_train_sm = X
X_train_sm = sm.add_constant(X_train_sm)

lm_sm = sm.OLS(Y,X_train_sm.astype(float)).fit()

# lm_sm.params
print(lm_sm.summary())
#Separating the data into response and regressors.
X1=test_new.drop(['PRICE1'],axis=1)
Y1=test_new['PRICE1']
X1=X1.reset_index();
#X.head()
Y1=Y1.reset_index();
#Y.head()
Y1=Y1.drop('index',axis=1);
#Y.head()
X1=X1.drop('index',axis=1);
#list(X.columns.values)
X1_sm = X1
X1_sm = sm.add_constant(X1_sm)
y_pred = lm_sm.predict(X1_sm)
# y_pred
res = Y1["PRICE1"] - y_pred
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y1["PRICE1"], y_pred)
r_squared = r2_score(y_pred,Y1["PRICE1"])
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
##Ridge Regression
from sklearn.linear_model import RidgeCV
clf = RidgeCV(alphas = np.arange(0.001,2,0.01),store_cv_values=True, cv=None)
model = clf.fit(X_train_sm, Y)
clf.score(X_train_sm,Y, sample_weight=None)
model.alpha_
clf.score(X1_sm,Y1,sample_weight=None)

y_pr = clf.predict(X1_sm)
y_pr = y_pr.tolist()
from itertools import chain
y_pred = list(chain(*y_pr))
y_pred = np.array(y_pred)
y_pred = pd.Series(y_pred)
res1 = Y1["PRICE1"] - y_pred
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y1["PRICE1"], y_pred)
r_squared = r2_score(y_pred,Y1["PRICE1"])
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
sns.residplot(res1,y_pred)
rf = RandomForestRegressor(n_estimators = 300, max_features=5)
rf.fit(X,np.array(Y))
preds = rf.predict(X1)
mse = mean_squared_error(Y1["PRICE1"], preds)
r_squared = r2_score(preds,Y1["PRICE1"])
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
xg_reg = xgb.XGBRegressor(objective ='reg:linear',max_depth = 5, min_child_weight = 1, colsample_bytree = 0.8)
X.head()
xg_reg.fit(X,Y)
preds = xg_reg.predict(X1)
mse = mean_squared_error(Y1["PRICE1"], preds)
r_squared = r2_score(preds,Y1["PRICE1"])
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
#Parameter tuning in Random Forest using Grid Search 
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_features': [3, 4, 5,],
    'n_estimators': [50, 100, 200, 300]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3)
# Fit the grid search to the data
grid_search.fit(X,Y)
grid_search.best_params_
# Hence we get the optimal parameters as: max features as 4 and n estimators as 200 
rf = RandomForestRegressor(n_estimators = 200, max_features=4)
rf.fit(X,np.array(Y))
preds = rf.predict(X1)
mse = mean_squared_error(Y1["PRICE1"], preds)
r_squared = r2_score(preds,Y1["PRICE1"])
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
#The grid search gave us similar results to our original parameters. Indicating that we need to expand 
#the range of parameters in the grid and apply ranges for more parameters. 
def xgboostcv(
              learning_rate,
              colsample_bytree,
              silent=True,
              nthread=-1):
    score= cross_val_score(xgb.XGBRegressor(objective='reg:linear',
                                             learning_rate=learning_rate,
                                             colsample_bytree=colsample_bytree),
                           X,
                           Y,
                           cv=5).mean()
    score=np.array(score)
    return score
xgboostBO = BayesianOptimization(xgboostcv,
                                 {
                                  'learning_rate': (0.01, 0.3),
                                  'colsample_bytree' :(0.5, 0.99)
                                  })
xgboostBO.maximize(init_points=3, n_iter=10)
xgb_bayesopt.res['max']['max_params']