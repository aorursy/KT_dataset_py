# Basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.simplefilter("ignore")

# Directry check

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Statistics library

from scipy.stats import norm

from scipy import stats

import scipy



# Data preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



# Visualization

from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns



# Dimension reduction

from sklearn.manifold import TSNE



# Machine learning

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

import lightgbm as lgb

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV



from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



# Validataion

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
# data loading

sample = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
# Create sale price log columns

df_train["log_SalePrice"] = np.log10(df_train["SalePrice"])
# dtype object

dtype = pd.DataFrame({"columns":df_train.dtypes.index,

                     "dtype":df_train.dtypes})

dtype["dtype"] = [str(i) for i in dtype["dtype"]]



# columns

num_columns = dtype.query('dtype=="int64" | dtype=="float64"')["columns"].values[1:-2]

obj_columns = dtype.query('dtype=="object"')["columns"].values



print("numerical_values_count:{}".format(len(num_columns)))

print("object_values_count:{}".format(len(obj_columns)))
### Separate data frame

y = df_train["log_SalePrice"]



df_num_train = df_train[num_columns]

df_num_test = df_test[num_columns]



df_cate_train = df_train[obj_columns]

df_cate_test = df_test[obj_columns]
# fill by median

columns = df_num_train.columns

for i in range(len(columns)):

    median = df_train[columns[i]].median()

    df_train[columns[i]].fillna(median, inplace=True)

    df_test[columns[i]].fillna(median, inplace=True)

    

df_num_train = df_train[num_columns]

df_num_test = df_test[num_columns]
# Training data

df_num_train["LotArea"] = np.log10(df_num_train["LotArea"]+1)

df_num_train["1stFlrSF"] = np.log10(df_num_train["1stFlrSF"]+1)

df_num_train["GrLivArea"] = np.log10(df_num_train["GrLivArea"]+1)



# test data

df_num_test["LotArea"] = np.log10(df_num_test["LotArea"]+1)

df_num_test["1stFlrSF"] = np.log10(df_num_test["1stFlrSF"]+1)

df_num_test["GrLivArea"] = np.log10(df_num_test["GrLivArea"]+1)
# prepairing dataframe

df_cha = pd.concat([df_cate_train,y], axis=1)



# define function

# Change to mean value function

def change_cate_mean(df1, df2):

    col_list_m = []

    for i in range(0,len(df1.columns)-1):

        # mean values

        index = df1.iloc[:,i].value_counts().index

        for j in range(len(index)):

            mean = df1[df1.iloc[:,i]==index[j]]["log_SalePrice"].mean()

            df1.iloc[:,i][df1.iloc[:,i]==index[j]] = mean

            df2.iloc[:,i][df2.iloc[:,i]==index[j]] = mean

        # update column name

        col = df1.columns[i]+'_m'

        col_list_m.append(col) 

    # create out put df

    df_c1 = df1.copy().drop("log_SalePrice", axis=1)

    df_c2 = df2.copy()

    df_c = pd.concat([df_c1, df_c2]).reset_index().drop("index", axis=1)

    df_c.columns = col_list_m

    return df_c



# Change to std value function

def change_cate_std(df1, df2):

    col_list_m = []

    for i in range(0,len(df1.columns)-1):

        # mean values

        index = df1.iloc[:,i].value_counts().index

        for j in range(len(index)):

            std = df1[df1.iloc[:,i]==index[j]]["log_SalePrice"].std()

            df1.iloc[:,i][df1.iloc[:,i]==index[j]] = std

            df2.iloc[:,i][df2.iloc[:,i]==index[j]] = std

        # update column name

        col = df1.columns[i]+'_s'

        col_list_m.append(col) 

    # create out put df

    df_c1 = df1.copy().drop("log_SalePrice", axis=1)

    df_c2 = df2.copy()

    df_c = pd.concat([df_c1, df_c2]).reset_index().drop("index", axis=1)

    df_c.columns = col_list_m

    return df_c



# Change to float value function

def change_float(df):

    for i in range(0,len(df.columns)-1):

        lists = []

        for j in range(len(df.iloc[:,i])):

            flo = float(df.iloc[:,i][j])

            lists.append(flo)

        df.iloc[:,i] = lists

    return df



# Fill by median function

def fill_median(df):

    for i in range(0,len(df.columns)-1):

        median = df.iloc[:,i].median()

        df.iloc[:,i].fillna(median, inplace=True)

    return df



# create df fix

df_cate_fix = pd.concat([

    fill_median(change_float(change_cate_mean(df_cha, df_cate_test))), fill_median(change_float(change_cate_std(df_cha, df_cate_test))) ], axis=1)



df_cate_fix.head()
# Re-Separate 

df_cate_train = df_cate_fix.head(len(df_cate_train))

df_cate_test = df_cate_fix.tail(len(df_cate_test)).reset_index().drop("index", axis=1)
# corrlation coefficient

coef = pd.concat([df_cate_train,y], axis=1).corr().round(3)



# Select variables with border of correlation with SalePrice values

# Check the accuracy(R2 score)

R2_train_score_list =[]

R2_test_score_list = []

variable_count_list = []

R2_valid_df = pd.DataFrame({})

# Roop

for i in range(0,8):

    col_select = coef[(coef["log_SalePrice"] > 0.1*i) | (coef["log_SalePrice"] < -0.1*i)].index

    # calc R2 score of Ridge regression

    # Dataset

    X = df_cate_train[col_select[:-1]]

    y = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    # Scaling

    msc = MinMaxScaler()

    msc.fit(X)

    X_train_msc = msc.fit_transform(X_train)

    X_test_msc = msc.fit_transform(X_test)

    # Training and score

    ridge = Ridge().fit(X_train_msc, y_train)

    R2_train_score = ridge.score(X_train_msc, y_train)

    R2_test_score = ridge.score(X_test_msc, y_test)

    variable_count = len(col_select)-1

    R2_train_score_list.append(R2_train_score)

    R2_test_score_list.append(R2_test_score)

    variable_count_list.append(variable_count)

    



# create df

R2_valid_df["valid"] = ["All",">0.1|<-0.1",">0.2|<-0.2",">0.3|<-0.3",">0.4|<-0.4",">0.5|<-0.5",">0.6|<-0.6",">0.7|<-0.7"]

R2_valid_df["R2_train_score"] = R2_train_score_list

R2_valid_df["R2_test_score"] = R2_test_score_list

R2_valid_df["R2_variable_count"] = variable_count_list



# Check

R2_valid_df
col_select = coef[(coef["log_SalePrice"] > 0.2) | (coef["log_SalePrice"] < -0.2)].index[:-1]
df_cate_train = df_cate_train[col_select]

df_cate_test = df_cate_test[col_select]
# Data set

X_data = pd.concat([df_num_train, df_cate_train], axis=1)

y = df_train["log_SalePrice"]

X_test = pd.concat([df_num_test, df_cate_test], axis=1)
# Correlation

matrix = X_data



# Visualization

plt.figure(figsize=(20,20))

hm = sns.heatmap(matrix.corr(), vmax=1, vmin=-1, cmap="bwr", square=True)
# quantile 75% sales price

quat_price_80 = y.quantile(0.80)

y_class = []

for i in y:

    if i >= quat_price_80:

        res = 1

        y_class.append(res)

    else:

        res = 0

        y_class.append(res)
# Train test data split

X_train, X_val, y_train, y_val = train_test_split(X_data, y_class, test_size=0.2, random_state=10)
# Random forest classifier

# Create instance

forest = RandomForestClassifier(n_estimators=10, random_state=10)



# Gridsearch

param_range = [10, 15, 20]

leaf = [70, 80, 90]

criterion = ["entropy", "gini", "error"]

param_grid = [{"n_estimators":param_range, "max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=1)



# Fitting

gs = gs.fit(X_train, y_train)



print(gs.best_score_)

print(gs.best_params_)
# Fitting for forest instance

forest = RandomForestClassifier(n_estimators=15, random_state=10,

                               criterion='gini', max_depth=15,

                               max_leaf_nodes=60)

forest.fit(X_train, y_train)



# Importance 

importance = forest.feature_importances_



# index

indices = np.argsort(importance)[::-1]



for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" %(f+1, 30, X_data.columns[indices[f]], importance[indices[f]]))
# Visualization

plt.figure(figsize=(15,6), facecolor="lavender")



plt.title('Feature Importances')

plt.bar(range(X_train.shape[1]), importance[indices], color='blue', align='center')

plt.xticks(range(X_train.shape[1]), X_data.columns[indices], rotation=90)

plt.xlim([-1, X_train.shape[1]])

plt.tight_layout()
# Select variables

number = 34

important_columns = X_data.columns[indices[0:number]]
# data set

# target price change to log form price distribution analysis

X = pd.concat([df_num_train, df_cate_train], axis=1)[important_columns]

y = df_train["log_SalePrice"]

X_test = pd.concat([df_num_test, df_cate_test], axis=1)[important_columns]
# Train test data split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
# Scaling

sc = StandardScaler()

sc.fit(X_train)

X_train_sc = sc.fit_transform(X_train)

X_val_sc = sc.fit_transform(X_val)

X_test_sc = sc.fit_transform(X_test)
# Create instance

tsne = TSNE(n_components=2, random_state=10)



# Figging

tsne.fit(X_train_sc)



# Fit transform

tsne_X_train = tsne.fit_transform(X_train_sc)

tsne_X_val = tsne.fit_transform(X_val_sc)
# Visualization by plot

x1 = tsne_X_train[:,0]

y1 = tsne_X_train[:,1]



x2 = tsne_X_val[:,0]

y2 = tsne_X_val[:,1]



plt.figure(figsize=(10,8))

plt.scatter(x1, y1, c="blue", label="train_data")

plt.scatter(x2, y2, c="red", label="val_data")

plt.xlabel("iso_axis1")

plt.ylabel("iso_axis2")

plt.legend()
# create instance

forest = RandomForestRegressor(n_estimators=100, random_state=10)



params = {"max_depth":[20,21,22,23, 24], "n_estimators":[31,32,33,34,35]}



# Fitting

cv_f = GridSearchCV(forest, params, cv = 10, n_jobs =10)



cv_f.fit(X_train, y_train)



print("Best params:{}".format(cv_f.best_params_))



best_f = cv_f.best_estimator_



# prediction

y_train_pred_f = best_f.predict(X_train)

y_val_pred_f = best_f.predict(X_val)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_f)))

print("MSE test;{}".format(mean_squared_error(y_val, y_val_pred_f)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_f)))

print("R2 score test:{}".format(r2_score(y_val, y_val_pred_f)))
# Create instance

xgbr = xgb.XGBRegressor()



params = {'learning_rate': [0.1, 0.15, 0.2, 0.25], 'max_depth': [3, 5,7,9], 

          'subsample': [0.8, 0.85, 0.9, 0.95], 'colsample_bytree': [0.3, 0.5, 0.8]}



# Fitting

cv_x = GridSearchCV(xgbr, params, cv = 10, n_jobs =1)

cv_x.fit(X_train, y_train)



print("Best params:{}".format(cv_x.best_params_))



best_x = cv_x.best_estimator_



# prediction

y_train_pred_x = best_x.predict(X_train)

y_val_pred_x = best_x.predict(X_val)



# prediction

y_train_pred_x = cv_x.predict(X_train)

y_val_pred_x = cv_x.predict(X_val)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_x)))

print("MSE val;{}".format(mean_squared_error(y_val, y_val_pred_x)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_x)))

print("R2 score val:{}".format(r2_score(y_val, y_val_pred_x)))
# Create instance

lgbm = lgb.LGBMRegressor()



params = {'learning_rate': [0.01,0.05, 0.08], 'max_depth': [10, 15, 20, 25]}



# Fitting

cv_lg = GridSearchCV(lgbm, params, cv = 10, n_jobs =1)

cv_lg.fit(X_train, y_train)



print("Best params:{}".format(cv_lg.best_params_))



best_lg = cv_lg.best_estimator_



# prediction

y_train_pred_lg = best_lg.predict(X_train)

y_val_pred_lg = best_lg.predict(X_val)



# prediction

y_train_pred_lg = cv_lg.predict(X_train)

y_val_pred_lg = cv_lg.predict(X_val)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_lg)))

print("MSE val;{}".format(mean_squared_error(y_val, y_val_pred_lg)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_lg)))

print("R2 score val:{}".format(r2_score(y_val, y_val_pred_lg)))
# Create instance

gbm = GradientBoostingRegressor(random_state=10)



params = {'max_depth': [2, 3, 5], 'learning_rate': [0.05, 0.1, 0.15, 0.2]}



# Fitting

cv_g = GridSearchCV(gbm, params, cv = 10, n_jobs =1)

cv_g.fit(X_train, y_train)



print("Best params:{}".format(cv_g.best_params_))



best_g = cv_g.best_estimator_



# prediction

y_train_pred_g = cv_g.predict(X_train)

y_val_pred_g = cv_g.predict(X_val)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_g)))

print("MSE val;{}".format(mean_squared_error(y_val, y_val_pred_g)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_g)))

print("R2 score val:{}".format(r2_score(y_val, y_val_pred_g)))
# Training and score

ridge = Ridge()

params = {'alpha': [1000, 100, 10, 1, 0.1, 0.01, 0.001]}



# Fitting

cv_r = GridSearchCV(ridge, params, cv = 10, n_jobs =1)

cv_r.fit(X_train_sc, y_train)



print("Best params:{}".format(cv_r.best_params_))



best_r = cv_r.best_estimator_



# prediction

y_train_pred_r = best_r.predict(X_train_sc)

y_val_pred_r = best_r.predict(X_val_sc)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_r)))

print("MSE val;{}".format(mean_squared_error(y_val, y_val_pred_r)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_r)))

print("R2 score val:{}".format(r2_score(y_val, y_val_pred_r)))
# Training and score

lasso = Lasso()

params = {'alpha': [1000, 100, 10, 1, 0.1, 0.01, 0.001]}



# Fitting

cv_l = GridSearchCV(lasso, params, cv = 10, n_jobs =1)

cv_l.fit(X_train_sc, y_train)





print("Best params:{}".format(cv_l.best_params_))



best_l = cv_l.best_estimator_



# prediction

y_train_pred_l = best_l.predict(X_train_sc)

y_val_pred_l = best_l.predict(X_val_sc)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_l)))

print("MSE val;{}".format(mean_squared_error(y_val, y_val_pred_l)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_l)))

print("R2 score val:{}".format(r2_score(y_val, y_val_pred_l)))
# Training and score

elas = ElasticNet()

params = {'alpha': [1000, 100, 10, 1, 0.1, 0.01, 0.001]}



# Fitting

cv_e = GridSearchCV(elas, params, cv = 10, n_jobs =1)

cv_e.fit(X_train_sc, y_train)



print("Best params:{}".format(cv_e.best_params_))



best_e = cv_e.best_estimator_



# prediction

y_train_pred_e = best_e.predict(X_train_sc)

y_val_pred_e = best_e.predict(X_val_sc)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_e)))

print("MSE val;{}".format(mean_squared_error(y_val, y_val_pred_e)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_e)))

print("R2 score val:{}".format(r2_score(y_val, y_val_pred_e)))
plt.figure(figsize=(10,6))

plt.scatter(y_val_pred_f, y_val_pred_f - y_val, c="blue", marker='o', alpha=0.5, label="RandomForest")

plt.scatter(y_val_pred_x, y_val_pred_x - y_val, c="red", marker='o', alpha=0.5, label="XGB")

plt.scatter(y_val_pred_g, y_val_pred_g - y_val, c="purple", marker='o', alpha=0.5, label="GBR")

plt.scatter(y_val_pred_lg, y_val_pred_lg - y_val, c="pink", marker='o', alpha=0.5, label="LGBM")

plt.scatter(y_val_pred_r, y_val_pred_r - y_val, c="green", marker='o', alpha=0.5, label="Rigde")

plt.scatter(y_val_pred_l, y_val_pred_l - y_val, c="orange", marker='o', alpha=0.5, label="Lasso")

plt.scatter(y_val_pred_e, y_val_pred_e - y_val, c="gray", marker='o', alpha=0.5, label="Elastic Net")



plt.xlabel('Predicted values')

plt.ylabel('Residuals')

plt.legend(loc = 'upper left')

plt.hlines(y = 0, xmin = 4.6, xmax = 5.7, lw = 2, color = 'black')

#plt.xlim([-10, 50])
# Create instance

tsne = TSNE(n_components=2, random_state=10)



# Figging

tsne.fit(X_train_sc)



# Fit transform

tsne_X_train = tsne.fit_transform(X_train_sc)

tsne_X_test = tsne.fit_transform(X_test_sc)



# Visualization by plot

x1 = tsne_X_train[:,0]

y1 = tsne_X_train[:,1]



x2 = tsne_X_test[:,0]

y2 = tsne_X_test[:,1]



plt.figure(figsize=(10,8))

plt.scatter(x1, y1, c="blue", label="train_data")

plt.scatter(x2, y2, c="green", label="test_data")

plt.xlabel("iso_axis1")

plt.ylabel("iso_axis2")

plt.legend()
## Test prediction

y_test_pred_f = best_f.predict(X_test)

y_test_pred_x = best_x.predict(X_test)

y_test_pred_lg = best_lg.predict(X_test)

y_test_pred_g = best_g.predict(X_test)

y_test_pred_r = best_r.predict(X_test_sc)

y_test_pred_l = best_l.predict(X_test_sc)

y_test_pred_e = best_e.predict(X_test_sc)



# submit prediction

y_submit = 10**((y_test_pred_f*0.05 + y_test_pred_x*0.05 + y_test_pred_lg*0.05 + y_test_pred_g*0.05 + y_test_pred_r*0.25 + y_test_pred_l*0.25 + y_test_pred_e*0.30))
# submit data

print("average:{}".format(round(y_submit.mean(),0)))

print("std:{}".format(round(y_submit.std(),0)))

print("max:{}".format(round(y_submit.max(),0)))

print("min:{}".format(round(y_submit.min(),0)))
# submit dataframe

submit = pd.DataFrame({"Id":df_test["Id"], "SalePrice":y_submit})
submit.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")