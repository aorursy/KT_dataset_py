import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import math

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns
df=pd.read_csv('../input/bike-sharing/Bike_Sharing.csv') 
df.head()
df.info()
df.describe()
df.dtypes
df.isnull().sum()
#changing to category datatype

col_cat = ["hour","weekday","month","season","weather","holiday","workingday"]

for var in col_cat:

    df[var] = df[var].astype("category")
df.dtypes
df  = df.drop(["index","date"],axis=1)
#plt.subplot(2,2,1)

plt.title('Temperature Vs Demand')

plt.scatter(df['temp'], df['demand'], c='b')
plt.title('atemp Vs Demand')

plt.scatter(df['atemp'], df['demand'], c='b')
sns.scatterplot(x="temp", y="atemp", data=df, hue="demand")

plt.show()
plt.title('Humidity Vs Demand')

plt.scatter(df['humidity'], df['demand'], c='b')
sns.scatterplot(x="windspeed", y="demand", data=df, hue="demand")

plt.show()
colors = ['g', 'r', 'm', 'b']

plt.title('Average Demand per Season')

cat_list = df['season'].unique()

cat_average = df.groupby('season').mean()['demand']

plt.bar(cat_list, cat_average, color=colors)
colors = ['g', 'r', 'm', 'b']

plt.title('Average Demand per month')

cat_list = df['month'].unique()

cat_average = df.groupby('month').mean()['demand']

plt.bar(cat_list, cat_average, color=colors)
plt.title('Average Demand per Holiday')

cat_list = df['holiday'].unique()

cat_average = df.groupby('holiday').mean()['demand']

plt.bar(cat_list, cat_average, color=colors)
plt.title('Average Demand per Weekday')

cat_list = df['weekday'].unique()

cat_average = df.groupby('weekday').mean()['demand']

plt.bar(cat_list, cat_average, color=colors)
plt.title('Average Demand per Year')

cat_list = df['year'].unique()

cat_average = df.groupby('year').mean()['demand']

plt.bar(cat_list, cat_average, color=colors)
plt.title('Average Demand per hour')

cat_list = df['hour'].unique()

cat_average = df.groupby('hour').mean()['demand']

plt.bar(cat_list, cat_average, color=colors)
plt.title('Average Demand per Workingday')

cat_list = df['workingday'].unique()

cat_average = df.groupby('workingday').mean()['demand']

plt.bar(cat_list, cat_average, color=colors)
plt.title('Average Demand per Weather')

cat_list = df['weather'].unique()

cat_average = df.groupby('weather').mean()['demand']

plt.bar(cat_list, cat_average, color=colors)
sns.set_style('darkgrid')

sns.distplot(df['demand'], bins = 100, color = 'blue')
#Q-Q Plot

from scipy import stats

plt = stats.probplot(df['demand'], plot=sns.mpl.pyplot)
sns.boxplot(x = 'demand', data = df, color = 'blue')
#Calculating the number of outliers

Q1 = df['demand'].quantile(0.25)

Q3 = df['demand'].quantile(0.75)

IQR = Q3 - Q1

outliers = df[(df['demand'] < (Q1 - 1.5 * IQR)) | (df['demand'] > (Q3 + 1.5 * IQR))]

print((len(outliers)/len(df))*100)
df_final = df[np.abs(df["demand"]-df["demand"].mean())<=(3*df["demand"].std())]

print ("Shape Of The Before Ouliers: ",df.shape)

print ("Shape Of The After Ouliers: ",df_final.shape)
tc = df.corr()

sns.heatmap(tc, annot = True, cmap = 'coolwarm')
df_final = df_final.drop(['weekday', 'year', 'workingday', 'atemp','casual', 'registered'], axis=1)
import matplotlib.pyplot as plt

# Autocorrelation of demand using acor

dff1 = pd.to_numeric(df_final['demand'], downcast='float')

plt.acorr(dff1, maxlags=12)
fig,axes = plt.subplots(ncols=2,nrows=2)

fig.set_size_inches(12, 10)

sns.distplot(df_final["demand"],ax=axes[0][0])

stats.probplot(df_final["demand"], dist='norm', fit=True, plot=axes[0][1])

sns.distplot(np.log(df_final["demand"]),ax=axes[1][0])

stats.probplot(np.log1p(df_final["demand"]), dist='norm', fit=True, plot=axes[1][1])
df_final['demand'] = np.log(df_final['demand'])
# Solve the problem of Autocorrelation

# Shift the demand by 3 lags



t_1 = df_final['demand'].shift(+1).to_frame()

t_1.columns = ['t-1']



t_2 = df_final['demand'].shift(+2).to_frame()

t_2.columns = ['t-2']



t_3 = df_final['demand'].shift(+3).to_frame()

t_3.columns = ['t-3']



df_final_lag = pd.concat([df_final, t_1, t_2, t_3], axis=1)
df_final_lag.head()
df_final_lag = df_final_lag.dropna()
df.columns
df_final_lag['windspeed'].value_counts()
from sklearn.ensemble import RandomForestRegressor

df_Wind_0 = df_final_lag[df_final_lag["windspeed"]==0]

df_Wind_Not0 = df_final_lag[df_final_lag["windspeed"]!=0]

Columns = ["season","weather","humidity","month","temp"]

rf_model = RandomForestRegressor()

rf_model.fit(df_Wind_Not0[Columns],df_Wind_Not0["windspeed"])



wind0Values = rf_model.predict(X= df_Wind_0[Columns])

df_Wind_0["windspeed"] = wind0Values

data = df_Wind_Not0.append(df_Wind_0)

data.reset_index(inplace=True)

data.drop('index',inplace=True,axis=1)
data.dtypes
data = pd.get_dummies(data, drop_first=True)
data.columns
data.shape
X = np.array(data.loc[:,data.columns!='demand'])

Y = np.array(data.loc[:,data.columns=='demand'])
print(X.shape)

print(Y.shape)
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn import metrics
def regression(X, Y, reg, param_grid, test_size=0.20):

    

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

      

    

    reg = RandomizedSearchCV(reg,parameters, cv = 10,refit = True)

    reg.fit(X_train, Y_train)     



    return X_train, X_test, Y_train, Y_test, reg
def evaluation_metrics(X_train, X_test, Y_train, Y_test, reg):

    Y_pred_train = reg.best_estimator_.predict(X_train)

    Y_pred_test = reg.best_estimator_.predict(X_test)

    

    print("Best Parameters:",reg.best_params_)

    print('\n')

    print("Mean cross-validated score of the best_estimator : ", reg.best_score_) 

    print('\n')

    MAE_train = metrics.mean_absolute_error(Y_train, Y_pred_train)

    MAE_test = metrics.mean_absolute_error(Y_test, Y_pred_test)

    print('MAE for training set is {}'.format(MAE_train))

    print('MAE for test set is {}'.format(MAE_test))

    print('\n')

    MSE_train = metrics.mean_squared_error(Y_train, Y_pred_train)

    MSE_test = metrics.mean_squared_error(Y_test, Y_pred_test)

    print('MSE for training set is {}'.format(MSE_train))

    print('MSE for test set is {}'.format(MSE_test))

    print('\n')

    RMSE_train = np.sqrt(metrics.mean_squared_error(Y_train, Y_pred_train))

    RMSE_test = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_test))

    print('RMSE for training set is {}'.format(RMSE_train))

    print('RMSE for test set is {}'.format(RMSE_test))

    print('\n')

    r2_train = metrics.r2_score(Y_train, Y_pred_train)

    r2_test = metrics.r2_score(Y_test, Y_pred_test)

    print("R2 value for train: ", r2_train)

    print("R2 value for test: ", r2_test)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

parameters = {'fit_intercept':[True,False],'normalize':[True,False], 'copy_X':[True, False]}

X_train, X_test, Y_train, Y_test, linreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)

evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = linreg)
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()

parameters = {'max_depth':[5,6,7,8,9,10]}

X_train, X_test, Y_train, Y_test, DTreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)

evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = DTreg)
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_jobs=-1)

parameters = {'n_estimators':[10,15,20,25],'max_depth':[5,6,7,8,9,10]}

X_train, X_test, Y_train, Y_test, RFreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)

evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = RFreg)
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor()

parameters = {'alpha':[0.01,0.001,0.0001],'n_estimators':[100,150,200],'max_depth':[3,5,7]}

X_train, X_test, Y_train, Y_test, XGreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)

evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = XGreg)
from sklearn.svm import SVR
reg = SVR()

parameters = {'max_iter':[1000,5000,10000]}

X_train, X_test, Y_train, Y_test, SVRreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)

evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = SVRreg)
from sklearn.neural_network import MLPRegressor
reg = MLPRegressor(activation='tanh',early_stopping=True)

parameters = {'solver':['sgd', 'adam'],'learning_rate_init':[0.01,0.001,0.0001],'hidden_layer_sizes':[10,25,50],'max_iter':[500,1000]}

X_train, X_test, Y_train, Y_test, MLPreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)

evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = MLPreg)
MLPreg.best_estimator_
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
Y_Pred_test = MLPreg.best_estimator_.predict(X_test)
fig, ax = plt.subplots(figsize=(12,7))

ax.scatter(Y_test, Y_Pred_test, edgecolors=(0, 0, 0))

ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k-', lw=4)

ax.set_xlabel('Actual')

ax.set_ylabel('Predicted')

ax.set_title("Ground Truth vs Predicted")

plt.show()