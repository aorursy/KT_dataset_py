import numpy as np #array for numerical processing

import pandas as pd #dataframe



import os

import random

import gc



from scipy import stats as stats # statistical tools



# OLS and  othe regression tools

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF



# metrics, preprocessing, model selection

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error

from sklearn.preprocessing import scale, StandardScaler, QuantileTransformer, RobustScaler

from sklearn.linear_model import LinearRegression, Lasso, Ridge



# for plots

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
def feature_table(df, description=None):

    '''

    creates a table summarising the features



    df: a pandas dataframe

    description: list having description of the variables

    '''

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values

    summary['Approx Missing %'] = 100*summary['Missing']//df.shape[0]

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.iloc[0].values

    summary['Second Value'] = df.iloc[1].values

    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    

    if (description!=None): summary['Description'] = pd.Series(description)



    return summary
def homoscedasticity_test(model):

    '''

    Function for testing the homoscedasticity of residuals in a linear regression model.

    It runs Breusch-Pagan

    

    Args:

    * model - fitted OLS model from statsmodels

    '''

    fitted_vals = model.predict()

    resids = model.resid





    bp_test = pd.DataFrame(sm.stats.het_breuschpagan(resids, model.model.exog), 

                           columns=['value'],

                           index=['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'])



    gq_test = pd.DataFrame(sm.stats.het_goldfeldquandt(resids, model.model.exog)[:-1],

                           columns=['value'],

                           index=['F statistic', 'p-value'])



    print('\n Breusch-Pagan test ----')

    print(bp_test)

    print('\n Goldfeld-Quandt test ----')

    print(gq_test)

    

def homoscedasticity_test2(y_true,y_predicted, exog):

    '''

    Function for testing the homoscedasticity of residuals in a linear regression model.

    It runs Breusch-Pagan

  

    '''

    temp1 = np.array(y_true).reshape(-1,1)

    temp2 = np.array(y_predicted).reshape(-1,1)



    resids = np.subtract(temp1,temp2) #calculating residuals



    bp_test = pd.DataFrame(sm.stats.het_breuschpagan(resids, exog), 

                           columns=['value'],

                           index=['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'])



    print('\n Breusch-Pagan test ----')

    return(bp_test)
from scipy import stats



def normality(variable):

    '''

    Function for running 4 statistical tests to investigate the normality of residuals.

    

    Argument:

    * model - fitted OLS models from statsmodels

    '''



    jb = stats.jarque_bera(variable)

    sw = stats.shapiro(variable)

    ad = stats.anderson(variable, dist='norm')

    ks = stats.kstest(variable, 'norm')

    

    print(f'Jarque-Bera test ---- statistic: {jb[0]:.4f}, p-value: {jb[1]}')

    print(f'Shapiro-Wilk test ---- statistic: {sw[0]:.4f}, p-value: {sw[1]:.4f}')

    print(f'Kolmogorov-Smirnov test ---- statistic: {ks.statistic:.4f}, p-value: {ks.pvalue:.4f}')

    print(f'Anderson-Darling test ---- statistic: {ad.statistic:.4f}, 5% critical value: {ad.critical_values[2]:.4f}')

    print('If the returned AD statistic is larger than the critical value, then for the 5% significance level, the null hypothesis that the data come from the Normal distribution should be rejected. ')
def regress(features, X_train, y_train, X_test, y_test):

    '''

    This function performs regression for a given feature set of size 'r' using training values

    It then calculates the RSS (Residual Sum of Squares) for the obtained model and returns the model as well as the RSS

    '''

    # Fit model on feature_set and calculate RSS

    model = sm.OLS(y_train,X_train[list(features)])

    regression = model.fit()

    RSS = ((regression.predict(X_test[list(features)]) - y_test) ** 2).sum()

    return {"model":regression, "RSS":RSS}

    #returning a dictionary with two key-value pairs

    #the key "model" has the statsmodels object as value

    #the key "RSS" has the Residual Sum of Squares as value
def forward(predictors, X_train, y_train, X_test, y_test):

    

    results = []

    remaining_predictors = [] # remaining_predictors consists of predictors that have not been used for model building

    for p in X_train.columns:

      if p not in predictors:

        remaining_predictors.append(p)

    

    # the predictors which have not been used for model building are added to the existing model, one at a time

    # the new model is then subjected to regression

    # the "results" list is a list of dictionaries, with each dictionary having the r-sized model and corresponding "RSS"

    for p in remaining_predictors:

        results.append(regress(predictors+[p], X_train, y_train, X_test, y_test))

    

    # converting the list of dictionaries to a dataframe

    models = pd.DataFrame(results)

    

    # out of the given set of r-sized models, we choose the model with the lowest RSS

    # this is equivalent to choosing the model with the highest R-square

    # models['RSS'].idxmin() returns the row index of the row having the lowest RSS value

    # models.loc[models['RSS'].idxmin()] chooses the row having the lowest RSS value

    # this corresponds to choosing a PandasSeries which has the "model" and the corresponding "RSS" value as elements

    best_model = models.loc[models["RSS"].idxmin()]

        

    # Return the best model

    return best_model
def cv(y,X,k):

  # function to perform cross-validation

  #k is the number of folds

  folds = np.random.choice(k, size = len(y), replace = True)



  # creating a DataFrame for storing the cross-validation errors

  cv_errors = pd.DataFrame(columns=range(1,k+1), index=range(1,(len(X.columns)+1)))

  cv_errors = cv_errors.fillna(0) #initializing the DataFrame with 0

  

  # creating a dataframe for storing the best models of different sizes

  models_cv = pd.DataFrame(columns=["RSS", "model"])

  

  # Outer loop iterates over all folds

  for j in range(1,k+1):



      # Reset predictors

      predictors = []

      print("FOLD:", j)

    

      # Inner loop iterates over each size i

      for i in range(1,len(X.columns)+1):    

    

          # The perform forward selection on the full dataset minus the jth fold, test on jth fold

          #models_cv.loc[i] = forward(predictors, X[folds != (j-1)], y[folds != (j-1)]["cnt"], X[folds == (j-1)], y[folds == (j-1)]["cnt"])

          models_cv.loc[i] = forward(predictors, X[folds != (j-1)], y[folds != (j-1)], X[folds == (j-1)], y[folds == (j-1)])

        

          # Save the cross-validated error for this fold

          cv_errors[j][i] = models_cv.loc[i]["RSS"]



          # Extract the predictors

          predictors = models_cv.loc[i]["model"].model.exog_names

          print("R2 of model with feature size ", len(predictors)," is: ", models_cv.loc[i]["model"].rsquared)

          

      print("-------------------------------------------------------------------------------")

  return cv_errors

        
def execute_CV(y,X,k):

  # function for finding the mean cross validation approach for each of the best models

  cv_errors = cv(y,X,k)

  print("CV_Errors Matrix:")

  print(cv_errors)

  print()

  cv_mean = cv_errors.apply(np.mean, axis=1)

  print("Mean values of cross validated errors for each feature:\n", cv_mean)

  return cv_mean
def deterministic(SEED=1729):

  np.random.seed(SEED)

  random.seed(SEED)

  os.environ['PYTHONHASHSEED']= str(SEED)
def reg_metrics (y_true,y_predicted):

  mse = mean_squared_error(y_true,y_predicted)

  print("MSE = " + str(mse))

  print("RMSE =" + str(np.sqrt(mse)))

  print("Mean Absolute Error =" + str(mean_absolute_error(y_true,y_predicted)))

  print('Median Absolute Error  = ' + str(median_absolute_error(y_true, y_predicted)))

  print('R^2                    = ' + str(r2_score(y_true, y_predicted)))
def resid_plot(y_true, y_predicted):

    '''

    1. Plots residual plots by taking the y_true and y_predicted as input.

    2. Can be used with any regression method be it linear, ridge, lasso, boosting-based, etc.

    '''

    temp1 = np.array(y_true).reshape(-1,1)

    temp2 = np.array(y_predicted).reshape(-1,1)



    resids = np.subtract(temp1,temp2) #calculating residuals



    sns.regplot(x=y_predicted, y=resids, fit_reg=True)



    plt.title("Residuals vs. Predicted Values")

    plt.xlabel("Predicted Values")

    plt.ylabel("Residuals")



    plt.show()

    plt.close()

    del temp1, temp2

    gc.collect()
def resids_hist(y_true,y_predicted):

  '''

  Plots histogram of residuals

  '''



  temp1 = np.array(y_true).reshape(-1,1)

  temp2 = np.array(y_predicted).reshape(-1,1)



  resids = np.subtract(temp1,temp2) 



  sns.distplot(resids)



  plt.title("Histogram of Residual Values")

  plt.xlabel("Residuals")

  plt.ylabel("Count")

  plt.show()

  del temp1, temp2

  gc.collect()
def resids_qq(y_true, y_predicted):

    resids = np.subtract(np.array(y_true).reshape(-1,1),  np.array(y_predicted).reshape(-1,1))

    stats.probplot(resids.flatten(), plot = plt)

    plt.title('Q-Q Plot of residuals')

    plt.xlabel('Quantiles of standard Normal distribution')

    plt.ylabel('Quantiles of residuals')

    plt.show()
def variance_inflation(df):

  vif = pd.DataFrame()

  vif["VIF Factor"] = [VIF(df.values, i) for i in range(df.shape[1])]

  vif["features"] = df.columns

  return vif
def plot_regularization(l, train_RMSE, test_RMSE, coefs, min_idx, title):   

    plt.plot(l, test_RMSE, color = 'green', label = 'Test RMSE')

    plt.plot(l, train_RMSE, label = 'Train RMSE')    

    plt.axvline(min_idx, color = 'black', linestyle = '--')

    plt.legend()

    plt.xlabel('Regularization parameter')

    plt.ylabel('Root Mean Square Error')

    plt.title(title)

    plt.show()

    

    plt.plot(l, coefs)

    plt.axvline(min_idx, color = 'black', linestyle = '--')

    plt.title('Model coefficient values \n vs. regularizaton parameter')

    plt.xlabel('Regularization parameter')

    plt.ylabel('Model coefficient value')

    plt.show()
def test_regularization_l2(x_train, y_train, x_test, y_test, l2):

    train_RMSE = []

    test_RMSE = []

    coefs = []

    for reg in l2:

        lin_mod = Ridge(alpha = reg)

        lin_mod.fit(x_train, y_train)

        coefs.append(lin_mod.coef_)

        y_score_train = lin_mod.predict(x_train)

        train_RMSE.append(mean_squared_error(y_train, y_score_train)**0.5)

        y_score = lin_mod.predict(x_test)

        test_RMSE.append(mean_squared_error(y_test, y_score)**0.5)

    min_idx = np.argmin(test_RMSE)

    min_l2 = l2[min_idx]

    min_RMSE = test_RMSE[min_idx] 

    

    title = 'Train and test root mean square error \n vs. regularization parameter'

    plot_regularization(l2, train_RMSE, test_RMSE, coefs, min_l2, title)

    return min_l2, min_RMSE
def test_regularization_l1(x_train, y_train, x_test, y_test, l1):

    train_RMSE = []

    test_RMSE = []

    coefs = []

    for reg in l1:

        lin_mod = Lasso(alpha = reg)

        lin_mod.fit(x_train, y_train)

        coefs.append(lin_mod.coef_)

        y_score_train = lin_mod.predict(x_train)

        train_RMSE.append(mean_squared_error(y_train, y_score_train)**0.5)

        y_score = lin_mod.predict(x_test)

        test_RMSE.append(mean_squared_error(y_test, y_score)**0.5)

    min_idx = np.argmin(test_RMSE)

    min_l1 = l1[min_idx]

    min_RMSE = test_RMSE[min_idx]

    

    title = 'Train and test root mean square error \n vs. regularization parameter'

    plot_regularization(l1, train_RMSE, test_RMSE, coefs, min_l1, title)

    return min_l1, min_RMSE
# Seeding random state to ensure reproducibility of results

deterministic(1729)
# reading data



data = pd.read_csv("/kaggle/input/multiple-linear-regression-dataset/MLRdata.csv")
data.head()
data.shape
len(data['Unnamed: 0'].unique())
len(data['V2'].unique())
constt = data['V2']
data.index = data['Unnamed: 0']

data.drop(['Unnamed: 0','V2'], axis=1, inplace=True)
data.head()
feature_table(data)
sorted(data['V3'].unique())
sorted(data['V4'].unique())
y = data['V1']

X = data.drop(['V1'],axis=1)
# creating training and validation sets



X_train, X_val, y_train, y_val = train_test_split(X,y,train_size = 0.85)
feature_table(X_train)
X_train.head()
plt.figure(figsize=(10,10))

normality(y_train)

sns.distplot(y_train)

plt.show()

plt.close()
plt.figure(figsize=(10,10))

sns.countplot(data['V3'])

plt.show()

plt.close()
plt.figure(figsize=(10,10))

sns.countplot(data['V4'])

plt.show()

plt.close()
plt.figure(figsize=(10,10))

normality(X_train['V5'])

sns.distplot(X_train['V5'])

plt.show()

plt.close()
plt.figure(figsize=(10,10))

normality(X_train['V6'])

sns.distplot(X_train['V6'])

plt.show()

plt.close()
plt.figure(figsize=(10,10))

normality(X_train['V7'])

sns.distplot(X_train['V7'])

plt.show()

plt.close()
plt.figure(figsize=(10,10))

normality(X_train['V8'])

sns.distplot(X_train['V8'])

plt.show()

plt.close()
corr = X_train.corr()

corr
plt.figure(figsize=(10,10))

sns.heatmap(corr, annot=True)

plt.show()

plt.close()
train = X_train.copy()

train['V1'] = y_train

#train.head()



val = X_val.copy()

val['V1'] = y_val

#val.head()



cols = train.columns
scaler = RobustScaler()



train = pd.DataFrame(scaler.fit_transform(train))

val = pd.DataFrame(scaler.transform(val))

train.columns = cols

val.columns = cols



y_train = train['V1']

X_train = train.drop(['V1'],axis=1)



y_val = val['V1']

X_val = val.drop(['V1'],axis=1)
X_train.head()
y_train, _ = stats.yeojohnson(y_train)
y_val, _ = stats.yeojohnson(y_val)
temp = X_train.copy()

temp = sm.add_constant(temp) #adding a constant term
temp.head()
model = sm.OLS(y_train, temp)

results = model.fit()

print(results.summary())
# Mean of residuals

results.resid.mean()
# Fitted parameters

results.params
temp2 = X_val.copy()

temp2 = sm.add_constant(temp2)

temp2.columns
features = temp.columns

preds = model.predict(results.params,exog=temp2[features])

reg_metrics(y_val,preds)

resids_hist(np.array(y_val), np.array(preds))

resids_qq(np.array(y_val), np.array(preds)) 

resid_plot(np.array(y_val), np.array(preds))

normality(results.resid)
homoscedasticity_test(results)
from scipy.stats.stats import pearsonr



for column in X_train.columns:

    corr_test = pearsonr(X_train[column], results.resid)

    print(f'Variable: {column} --- correlation: {corr_test[0]:.4f}, p-value: {corr_test[1]:.4f}')
X_train.apply(np.var, axis=0)
variance_inflation(temp)
temp = X_train.drop(['V6'],axis=1).copy()

temp = sm.add_constant(temp)
temp.head()
model = sm.OLS(y_train, temp)

results = model.fit()

print(results.summary())
results.resid.mean()
results.params
temp2 = X_val.copy()

temp2 = sm.add_constant(temp2)

temp2.columns
features = temp.columns

preds = model.predict(results.params,exog=temp2[features])

reg_metrics(y_val,preds)

resids_hist(np.array(y_val), np.array(preds))

resids_qq(np.array(y_val), np.array(preds)) 

resid_plot(np.array(y_val), np.array(preds))

normality(results.resid)
homoscedasticity_test(results)
variance_inflation(temp)
temp = X_train.drop(['V6','V5'],axis=1).copy()

temp = sm.add_constant(temp)
temp.head()
model = sm.OLS(y_train, temp)

results = model.fit()

print(results.summary())
results.resid.mean()
results.params
temp2 = X_val.copy()

temp2 = sm.add_constant(temp2)

temp2.columns
features = temp.columns

preds = model.predict(results.params,exog=temp2[features])

reg_metrics(y_val,preds)

resids_hist(np.array(y_val), np.array(preds))

resids_qq(np.array(y_val), np.array(preds)) 

resid_plot(np.array(y_val), np.array(preds))

normality(results.resid)
homoscedasticity_test(results)
variance_inflation(temp)
# Finding the best size of the model



cv_mean_day = execute_CV(y_train, X_train,10)

plt.figure(figsize=(15,10))

plt.plot(cv_mean_day)

plt.xlabel('# Predictors')

plt.ylabel('CV Error')

plt.title("CV Error vs. Model Size for Daily Data")

plt.plot(cv_mean_day.idxmin(), cv_mean_day.min(), "or")
# Best features

predictors = []

for i in range(1,6):

  best_ = forward(predictors, X_train, y_train, X_val, y_val) #we pass the full dataset

  predictors = best_[0].model.exog_names

print(best_[0].summary())

print("Best features:", best_[0].model.exog_names)

#print("R-squared:", best_[0].rsquared)

#print("Residuals:\n", best_[0].resid)





preds = best_[0].predict(exog=X_val[predictors])

reg_metrics(y_val,preds)

resids_hist(np.array(y_val), np.array(preds))

resids_qq(np.array(y_val), np.array(preds)) 

resid_plot(np.array(y_val), np.array(preds))
homoscedasticity_test2(y_val,preds,X_val)
variance_inflation(temp)
from sklearn.decomposition import PCA

from sklearn.cross_decomposition import PLSRegression, PLSSVD
pca = PCA()

X_reduced_train = pca.fit_transform(scale(X_train))
plt.figure(figsize=(15,10))

# 10-fold CV, with shuffle

n = len(X_reduced_train)

kf_10 = KFold(n_splits=10, shuffle=True, random_state=1)



regr = LinearRegression()

mse = []



# Calculate MSE with only the intercept (no principal components in regression)

score = -1*cross_val_score(regr, np.ones((n,1)), y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    

mse.append(score)



# Calculate MSE using CV for the principle components, adding one component at the time.

for i in np.arange(1,7):

    score = -1*cross_val_score(regr, X_reduced_train[:,:i], y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()

    mse.append(score)

    

# Plot results    

plt.plot(mse, '-v')

plt.xlabel('Number of principal components in regression')

plt.ylabel('MSE')

plt.title('V1')

plt.xlim(xmin=-1);
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
X_reduced_test = pca.transform(scale(X_val))[:,:5]



# Train regression model on training data 

regr = LinearRegression()

regr.fit(X_reduced_train[:,:5], y_train)
# Prediction with validation data

preds = regr.predict(X_reduced_test)

reg_metrics(y_val, preds)

resids_hist(np.array(y_val), np.array(preds))

resids_qq(np.array(y_val), np.array(preds)) 

resid_plot(np.array(y_val), np.array(preds))
homoscedasticity_test2(y_val,preds,X_val)
n = len(X_train)



# 10-fold CV, with shuffle

kf_10 = KFold(n_splits=10, shuffle=True, random_state=1)



mse = []



for i in np.arange(1, 5):

    pls = PLSRegression(n_components=i)

    score = cross_val_score(pls, scale(X_train), y_train, cv=kf_10, scoring='neg_mean_squared_error').mean()

    mse.append(-score)



# Plot results

plt.plot(np.arange(1, 5), np.array(mse), '-v')

plt.xlabel('Number of principal components in regression')

plt.ylabel('MSE')

plt.title('V1')

plt.xlim(xmin=-1)
pls = PLSRegression(n_components=2)

pls.fit(scale(X_train), y_train)
preds = pls.predict(scale(X_val))

reg_metrics(y_val, preds)

resids_hist(np.array(y_val), np.array(preds))

resids_qq(np.array(y_val), np.array(preds)) 

resid_plot(np.array(y_val), np.array(preds))
homoscedasticity_test2(y_val,preds,X_val)
huber_t = sm.RLM(y_train, X_train, M=sm.robust.norms.HuberT())



hub_results = huber_t.fit(cov="H2")

#print(hub_results.params)

#print(hub_results.bse)
print(hub_results.summary(yname='y',

            xname=['var_%d' % i for i in range(len(hub_results.params))]))
preds = hub_results.predict(X_val)



reg_metrics(y_val, preds)

resids_hist(np.array(y_val), np.array(preds))

resids_qq(np.array(y_val), np.array(preds)) 

resid_plot(np.array(y_val), np.array(preds))
homoscedasticity_test2(y_val,preds,X_val)
# Performing ridge regression

l2 = [x for x in range(1,101)]

out_l2 = test_regularization_l2(X_train, y_train, X_val, y_val, l2)

print(out_l2)
lin_mod_l2 = Ridge(alpha = out_l2[0])

lin_mod_l2.fit(X_train, y_train)

y_score_l2 = lin_mod_l2.predict(X_val)



reg_metrics(y_val, y_score_l2)

resids_hist(np.array(y_val), np.array(y_score_l2)) 

resids_qq(np.array(y_val), np.array(y_score_l2)) 

resid_plot(np.array(y_val), np.array(y_score_l2))
homoscedasticity_test2(y_val,preds,X_val)
l1 = [x/5000 for x in range(1,101)]

out_l1 = test_regularization_l1(X_train, y_train, X_val, y_val, l1)

print(out_l1)
lin_mod_l1 = Lasso(alpha = out_l1[0])

lin_mod_l1.fit(X_train, y_train)

y_score_l1 = lin_mod_l1.predict(X_val)



reg_metrics(y_val, y_score_l1) 

resids_hist(np.array(y_val), np.array(y_score_l1))

resids_qq(np.array(y_val), np.array(y_score_l1)) 

resid_plot(np.array(y_val), np.array(y_score_l1))
homoscedasticity_test2(y_val,preds,X_val)
from sklearn.tree import DecisionTreeRegressor

estimator = DecisionTreeRegressor()

param_dict = {'max_depth': [1,2,5,1000],

              'criterion': ['mse','mae','friedman_mse']

              }

dt = GridSearchCV(estimator,cv = KFold(10), param_grid=param_dict, n_jobs=-1)

dt.fit(X_train,y_train)

preds = dt.predict(X_val)
reg_metrics(y_val,preds)

resids_hist(np.array(y_val), np.array(preds))

resids_qq(np.array(y_val), np.array(preds)) 

resid_plot(np.array(y_val), np.array(preds))
dt.best_params_
from sklearn.ensemble import RandomForestRegressor

estimator = RandomForestRegressor()

param_dict = {'n_estimators': [100,1000,2000],

              'max_depth': [2,5,10]

              

}

rf = GridSearchCV(estimator,cv = KFold(10), param_grid=param_dict, n_jobs=-1)

rf.fit(X_train,y_train)
preds = rf.predict(X_val)

reg_metrics(y_val,preds)

resids_hist(np.array(y_val), np.array(preds))

resids_qq(np.array(y_val), np.array(preds)) 

resid_plot(np.array(y_val), np.array(preds))
rf.best_params_
import xgboost as xgb



estimator = xgb.XGBRegressor()

param_dict = {'n_estimators': [100,500],

               'max_depth': [2,5,10],

              'learning_rate':[0.001,0.01,0.1],

              

              

}



reg_xg = GridSearchCV(estimator,cv = KFold(10), param_grid=param_dict, n_jobs=-1)



reg_xg.fit(X_train,y_train)
preds = reg_xg.predict(X_val)



rmse = np.sqrt(mean_squared_error(y_val, preds))



reg_metrics(y_val,preds)

resids_hist(np.array(y_val), np.array(preds))

resids_qq(np.array(y_val), np.array(preds)) 

resid_plot(np.array(y_val), np.array(preds))
reg_xg.best_params_
from lightgbm import LGBMRegressor

reg_lgb = LGBMRegressor(objective='regression',num_leaves=6,

                              learning_rate=0.01,

                        n_iterations=8000,

                        n_estimators=4000) 

reg_lgb.fit(X_train,y_train)
preds = reg_lgb.predict(X_val)



rmse = np.sqrt(mean_squared_error(y_val, preds))





reg_metrics(y_val, preds) 

resids_hist(np.array(y_val), np.array(preds))

resids_qq(np.array(y_val), np.array(preds)) 

resid_plot(np.array(y_val), np.array(preds))
homoscedasticity_test2(y_val,preds,X_val)