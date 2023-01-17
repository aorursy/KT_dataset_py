# this may need to be installed separately with
# !pip install category-encoders
import category_encoders as ce

# python general
import pandas as pd
import numpy as np
from collections import OrderedDict


#scikit learn

import sklearn
from sklearn.base import clone

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# ML models
from sklearn import tree
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# error metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.preprocessing import StandardScaler
# plotting and display
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from IPython.display import display
pd.options.display.max_columns = None

# widgets and widgets based libraries
import ipywidgets as widgets
from ipywidgets import interact, interactive
def rmse(y_true, y_pred):
    res = np.sqrt(((y_true - y_pred) ** 2).mean())
    return res

def mape(y_true, y_pred):
    y_val = np.maximum(np.array(y_true), 1e-8)
    return (np.abs(y_true -y_pred)/y_val).mean()
metrics_dict_res = OrderedDict([
            ('mean_absolute_error', mean_absolute_error),
            ('median_absolute_error', median_absolute_error),
            ('root_mean_squared_error', rmse),
            ('mean abs perc error', mape)
            ])
def regression_metrics_yin(y_train, y_train_pred, y_test, y_test_pred,
                           metrics_dict, format_digits=None):
    df_results = pd.DataFrame()
    for metric, v in metrics_dict.items():
        df_results.at[metric, 'train'] = v(y_train, y_train_pred)
        df_results.at[metric, 'test'] = v(y_test, y_test_pred)

    if format_digits is not None:
        df_results = df_results.applymap(('{:,.%df}' % format_digits).format)

    return df_results
def describe_col(df, col):
    display(df[col].describe())

def val_count(df, col):
    display(df[col].value_counts())

def show_values(df, col):
    print("Number of unique values:", len(df[col].unique()))
    return display(df[col].value_counts(dropna=False))
def plot_distribution(df, col, bins=100, figsize=None, xlim=None, font=None, histtype='step'):
    if font is not None:
        mpl.rc('font', **font)

    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=(10, 6))
    dev = df[col]    
    dev.plot(kind='hist', bins=bins, density=True, histtype=histtype, color='b', lw=2,alpha=0.99)
    print('mean:', dev.mean())
    print('median:', dev.median())
    if xlim is not None:
        plt.xlim(xlim)
    return plt.gca()
def plot_feature_importances(model, feature_names=None, n_features=20):
    if feature_names is None:
        feature_names = range(n_features)
    
    importances = model.feature_importances_
    importances_rescaled = 100 * (importances / importances.max())
    xlabel = "Relative importance"

    sorted_idx = np.argsort(-importances_rescaled)

    names_sorted = [feature_names[k] for k in sorted_idx]
    importances_sorted = [importances_rescaled[k] for k in sorted_idx]

    pos = np.arange(n_features) + 0.5
    plt.barh(pos, importances_sorted[:n_features], align='center')

    plt.yticks(pos, names_sorted[:n_features])
    plt.xlabel(xlabel)

    plt.title("Feature importances")

    return plt.gca()
def plot_act_vs_pred(y_act, y_pred, scale=1, act_label='actual', pred_label='predicted', figsize=None, xlim=None,
                     ylim=None, font=None):
    
    if font is not None:
        mpl.rc('font', **font)

    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=(10, 6))
    plt.scatter(y_act/scale, y_pred/scale)
    x = np.linspace(0, y_act.max()/scale, 10)
    plt.plot(x, x)
    plt.xlabel(act_label)
    plt.ylabel(pred_label)
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim([0, 1e2])
    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim([0, 1e2])
    return plt.gca()
def compute_perc_deviation(y_act, y_pred, absolute=False):
    dev = (y_pred - y_act)/y_act * 100
    if absolute:
        dev = np.abs(dev)
        dev.name = 'abs % error'
    else:
        dev.name = '% error'
    return dev

def plot_dev_distribution(y_act, y_pred, absolute=False, bins=100, figsize=None, xlim=None, font=None):
    if font is not None:
        mpl.rc('font', **font)

    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=(10, 6))
    dev = compute_perc_deviation(y_act, y_pred, absolute=absolute)
    dev.plot(kind='hist', bins=bins, density=True)
    print('mean % dev:', dev.mean())
    print('median % dev:', dev.median())
    # plt.vlines(dev.mean(), 0, 0.05)
    plt.title('Distribution of errors')
    plt.xlabel('% deviation')
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim([-1e2, 1e2])
    return plt.gca()
categorical_features = [
    'Body_Type',
    'Driven_Wheels',
    'Global_Sales_Sub-Segment',
    'Brand',
    'Nameplate',
    'Transmission',
    'Turbo',
    'Fuel_Type',
    'PropSysDesign',
    'Plugin',
    'Registration_Type',
    'country_name'
]

numeric_features = [
    'Generation_Year',
    'Length',
    'Height',
    'Width',
    'Engine_KW',
    'No_of_Gears',
    'Curb_Weight',
    'CO2',
    'Fuel_cons_combined',
    'year'
]

all_numeric_features = list(numeric_features)
all_categorical_features = list(categorical_features)

target = [
    'Price_USD'
]

target_name = 'Price_USD'
ml_model_type = 'Linear Regression'
#ml_model_type = 'Decision Tree'
#ml_model_type = 'Random Forest'
#ml_model_type ='xg'

regression_metric = 'mean abs perc error'

do_grid_search_cv = True
scoring_greater_is_better = True  # THIS NEEDS TO BE SET CORRECTLY FOR CV GRID SEARCH

do_retrain_total = True
write_predictions_file = True

# relative size of test set
test_size = 0.2
random_state = 50
df = pd.read_csv('/kaggle/input/ihsmarkit-hackathon-june2020/train_data.csv',index_col='vehicle_id')
df['date'] = pd.to_datetime(df['date'])
# basic commands on a dataframe
# df.info()
df.head(5)
# df.shape
# df.head()
# df.tail()
df1 = df[df['Price_USD'] < 100000]
df=df1
# df
df.Brand = df.Brand.str.lower()

def replace_name(a,b):
    df.Brand.replace(a,b,inplace=True)

replace_name('hyundai-beijing','hyundai')
replace_name('youngman lotus','youngman')
replace_name('beijing auto','beijing')
replace_name('volkswagen-faw','volkswagen')
replace_name('beijing','beijing auto')
replace_name('piaggio','fiat')

df['country_name'].value_counts()
df.groupby(['year', 'country_name'])['date'].count()
df_oos = pd.read_csv('/kaggle/input/ihsmarkit-hackathon-june2020/oos_data.csv', index_col='vehicle_id')
df_oos['date'] = pd.to_datetime(df_oos['date'])
df_oos['year'] = df_oos['date'].map(lambda d: d.year)


def replace_name(a,b):
    df_oos.Brand.replace(a,b,inplace=True)

replace_name('hyundai-beijing','hyundai')
replace_name('youngman lotus','youngman')
replace_name('beijing auto','beijing')
replace_name('volkswagen-faw','volkswagen')
replace_name('beijing','beijing auto')
replace_name('piaggio','fiat')

# df_oos.shape
df_oos.head()
df_oos.groupby(['year', 'country_name'])['date'].count()
# unique values, categorical variables
for col in all_categorical_features:
    print(col, len(df[col].unique()))
interactive(lambda col: show_values(df, col), col=all_categorical_features)
# #before going for train test split, do standard scalar
# scalar = StandardScaler()
 
# # for column_name in ['Length','Height','Curb_Weight']:
    
# #     np_column = np.array(df[column_name])
# df['Length'] = scalar.fit_transform(np_column.reshape(-1, 1))
# df['Height'] = scalar.fit_transform(np_column.reshape(-1, 1))
# df['Curb_Weight'] = scalar.fit_transform(np_column.reshape(-1, 1))
# df['CO2'] = scalar.fit_transform(np_column.reshape(-1, 1))
# summary statistics
df.describe()
#df[numeric_features + target].describe()

figsize = (16,12)
sns.set(style='whitegrid', font_scale=2)

bins = 1000
bins = 40
#xlim = [0,100000]
xlim = None
price_mask = df['Price_USD'] <70000
interactive(lambda col: plot_distribution(df[price_mask], col, bins=bins, xlim=xlim), col=sorted(all_numeric_features + target))
#interactive(lambda col: plot_distribution(df, col, bins=bins, xlim=xlim), col=sorted(all_numeric_features + target))
# this is quite slow
sns.set(style='whitegrid', font_scale=1)
# sns.pairplot(df[numeric_features[:6] + target].iloc[:10000])
#sns.pairplot(df[['Engine_KW'] + target].iloc[:10000])
price_mask = df['Price_USD'] < 70000
df_temp = df[price_mask].copy()
sns.pairplot(df_temp[['Engine_KW'] + target])
additional_numeric_features = []
features_drop = []

categorical_features = [
    'Body_Type',
    'Driven_Wheels',
    'Global_Sales_Sub-Segment',
    'Brand',
    'Nameplate',
    'Transmission',
    'Turbo',
    'Fuel_Type',
    'PropSysDesign',
    'Plugin',
    'Registration_Type',
    'country_name'
]

numeric_features = [
    'Generation_Year',
    'Length',
    'Height',
    'Width',
    'Engine_KW',
    'No_of_Gears',
    'Curb_Weight',
    'CO2',
    'Fuel_cons_combined',
    'year'
]
feature_main=[]
feature_main=['Body_Type', 'Global_Sales_Sub-Segment','Brand', 'Length','Width','Engine_KW','No_of_Gears','year','Fuel_Type','Registration_Type','country_name','Transmission']
if ml_model_type == 'Linear Regression':
    features_drop = categorical_features + numeric_features
    features_to_use =feature_main
    for feature in features_to_use:
        features_drop.remove(feature)
else:
 
    features_drop = categorical_features + numeric_features
    features_to_use =feature_main
    for feature in features_to_use:
        features_drop.remove(feature)    #we are removing needed features from the features_drop
categorical_features = list(filter(lambda f: f not in features_drop, categorical_features))
numeric_features = list(filter(lambda f: f not in features_drop, numeric_features))

features = categorical_features + numeric_features + additional_numeric_features
model_columns = features + [target_name]

#dataframe for further processing
df_proc = df[model_columns].copy()
df_proc.shape

# One-hot encoding
encoder = ce.OneHotEncoder(cols=categorical_features , handle_unknown='value', 
                           use_cat_names=True)
 
encoder.fit(df_proc)
df_comb_ext = encoder.transform(df_proc)
features_ext = list(df_comb_ext.columns)
features_ext.remove(target_name)
#del df_proc
#df_comb_ext.head()
df_comb_ext.describe()
# df_comb_ext.memory_usage(deep=True).sum()/1e9
#features_model
df_comb_ext.shape
X_train, X_test, y_train, y_test = train_test_split(df_comb_ext[features_ext], df_comb_ext[target_name], 
                                                    test_size=test_size, random_state=random_state)

print(X_train.shape)
print(X_test.shape)
if ml_model_type == 'Linear Regression':
    model_hyper_parameters_dict = OrderedDict(fit_intercept=True, normalize=False)
    regressor =  LinearRegression(**model_hyper_parameters_dict)

if ml_model_type == 'Decision Tree':
    model_hyper_parameters_dict = OrderedDict(max_depth=1, random_state=random_state)
    regressor =  DecisionTreeRegressor(**model_hyper_parameters_dict)
      
if ml_model_type == 'Random Forest':

    model_hyper_parameters_dict = OrderedDict(n_estimators=10, 
                                              max_depth=4, 
                                              min_samples_split=2, 
                                              max_features='sqrt',
                                              min_samples_leaf=1, 
                                              random_state=random_state, 
                                              n_jobs=4)
   
    regressor = RandomForestRegressor(**model_hyper_parameters_dict)
        
base_regressor = clone(regressor)
    
# if do_grid_search_cv:
    
#     scoring = make_scorer(metrics_dict_res[regression_metric], greater_is_better=scoring_greater_is_better)
    
#     if ml_model_type == 'Random Forest':
        

#         grid_parameters = [{'n_estimators': [10], 'max_depth': [3, 5, 10], 
#                              'min_samples_split': [2,4], 'min_samples_leaf': [1]} ]
        
#     n_splits = 4
#     n_jobs = 4
# #     cv_regressor = GridSearchCV(regressor, grid_parameters, cv=n_splits, scoring=scoring, return_train_score=True,
# #                                 refit=True, n_jobs=n_jobs)    
#     cv_regressor
# do_grid_search_cv =False
# if do_grid_search_cv:
#     cv_regressor.fit(X_train, y_train)
#     regressor_best = cv_regressor.best_estimator_
#     model_hyper_parameters_dict = cv_regressor.best_params_
#     train_scores = cv_regressor.cv_results_['mean_train_score']
#     test_scores = cv_regressor.cv_results_['mean_test_score']
#     test_scores_std = cv_regressor.cv_results_['std_test_score']
#     cv_results = cv_regressor.cv_results_
# else:
#     regressor.fit(X_train, y_train)
# if do_grid_search_cv:
#     # print(cv_results)
#     print(model_hyper_parameters_dict)
#     plt.plot(-train_scores, label='train')
#     plt.plot(-test_scores, label='test')
#     plt.xlabel('Parameter set #')
#     plt.legend()
#     regressor = regressor_best
# y_train_pred = regressor.predict(X_train)
# y_test_pred = regressor.predict(X_test)
# if ml_model_type == 'Linear Regression':
#     df_reg_coef = (pd.DataFrame(zip(['intercept'] + list(X_train.columns), 
#                                [regressor.intercept_] + list(regressor.coef_)))
#                  .rename({0: 'feature', 1: 'coefficient value'}, axis=1))
#     display(df_reg_coef) 
#     df_reg_coef.to_csv("coefficent.csv")
# if hasattr(regressor, 'feature_importances_'):
#     sns.set(style='whitegrid', font_scale=1.5)
#     plt.figure(figsize=(12,10))
#     plot_feature_importances(regressor, features_ext, n_features=np.minimum(20, X_train.shape[1]))
# df_regression_metrics = regression_metrics_yin(y_train, y_train_pred, y_test, y_test_pred,
#                                                metrics_dict_res, format_digits=3)

# df_output = df_regression_metrics.copy()
# df_output.loc['Counts','train'] = len(y_train)
# df_output.loc['Counts','test'] = len(y_test)
# df_output

# figsize = (16,10)
# xlim = [0, 250]
# font={'size': 20}
# sns.set(style='whitegrid', font_scale=2.5)
# act_label = 'actual price [k$]'
# pred_label='predicted price [k$]'
# plot_act_vs_pred(y_test, y_test_pred, scale=1000, act_label=act_label, pred_label=pred_label, 
#                  figsize=figsize, xlim=xlim, ylim=xlim, font=font)
# print()
# figsize = (14,8)
# xlim = [0, 100]
# #xlim = [-100, 100]
# # xlim = [-50, 50]
# #xlim = [-20, 20]

# font={'size': 20}
# sns.set(style='whitegrid', font_scale=1.5)

# p_error = (y_test_pred - y_test)/y_test *100
# print(p_error)
# df_p_error = pd.DataFrame(p_error.values, columns=['percent_error'])

# #display(df_p_error['percent_error'].describe().to_frame())

# bins=1000
# bins=500
# #bins=100
# absolute = True
# #absolute = False
# plot_dev_distribution(y_test, y_test_pred, absolute=absolute, figsize=figsize, 
#                       xlim=xlim, bins=bins, font=font)
# print()
# from sklearn.svm import SVR

import xgboost as xgb
from sklearn.metrics import mean_squared_error
xg_reg = xgb.XGBRegressor()
# xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.8, learning_rate = 0.8,
#                max_depth = 6 , n_estimators = 10)
# xg_reg = xgb.XGBRegressor(objective ='reg:linear', 
#                max_depth =5  , n_estimators = 100 )

# num_boost_round=50,early_stopping_rounds=10,metrics="rmse"
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
y_train_pred = xg_reg.predict(X_train)
y_test_pred = xg_reg.predict(X_test)
y_train_pred
y_test_pred
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
if hasattr(xg_reg, 'feature_importances_'):
    sns.set(style='whitegrid', font_scale=1.5)
    plt.figure(figsize=(12,10))
    plot_feature_importances(xg_reg, features_ext, n_features=np.minimum(20, X_train.shape[1]))
from sklearn.metrics import accuracy_score
df_regression_metrics = regression_metrics_yin(y_train, y_train_pred, y_test, y_test_pred,
                                               metrics_dict_res, format_digits=3)

df_output = df_regression_metrics.copy()
df_output.loc['Counts','train'] = len(y_train)
df_output.loc['Counts','test'] = len(y_test)
df_output
figsize = (14,8)
xlim = [0, 100]
#xlim = [-100, 100]
# xlim = [-50, 50]
#xlim = [-20, 20]

font={'size': 20}
sns.set(style='whitegrid', font_scale=1.5)

p_error = (y_test_pred - y_test)/y_test *100
print(p_error)
df_p_error = pd.DataFrame(p_error.values, columns=['percent_error'])

#display(df_p_error['percent_error'].describe().to_frame())

bins=1000
bins=500
#bins=100
absolute = True
#absolute = False
plot_dev_distribution(y_test, y_test_pred, absolute=absolute, figsize=figsize, 
                      xlim=xlim, bins=bins, font=font)
print()
if do_retrain_total:
    cv_opt_model = clone(base_regressor.set_params(**model_hyper_parameters_dict))
    # train on complete data set
    X_train_full = df_comb_ext[features_ext].copy()
    y_train_full = df_comb_ext[target_name].values
    cv_opt_model.fit(X_train_full, y_train_full) 
    regressor = cv_opt_model
# df_oos.head()
df_proc_oos = df_oos[model_columns[:-1]].copy()
df_proc_oos[target_name] = 1
df_proc_oos.head()
df_comb_ext_oos = encoder.transform(df_proc_oos)

df_comb_ext_oos.drop(target_name, axis=1, inplace=True)
y_oos_pred = xg_reg.predict(df_comb_ext_oos)
y_oos_pred
print(y_oos_pred)
id_col = 'vehicle_id'
df_out = (pd.DataFrame(y_oos_pred, columns=[target_name], index=df_comb_ext_oos.index)
            .reset_index()
            .rename({'index': id_col}, axis=1))
df_out.head()
df_out.shape
if write_predictions_file:
    df_out.to_csv('XGBoost.csv', index=False)

