import pandas as pd

from pandas import Series

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from scipy.interpolate import interp1d



from sklearn import metrics

from sklearn.metrics import mean_squared_error, confusion_matrix

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV



import warnings

warnings.filterwarnings("ignore")
bakery_temp = pd.read_pickle('../input/bakery-temperature-summary/bakery_temp_sum_dataframe.pkl')
bakery_temp.info()
bakery_temp.head(2)
list_of_event = ['2016-11-05','2016-11-12','2017-01-28','2017-02-04','2017-02-18','2017-03-04']

is_bank_holiday = ['2016-12-25','2016-12-26','2017-01-01','2017-01-02']
dates = pd.DatetimeIndex(bakery_temp['Date_Time'])
#Create new dataset with only item and date 

bakery = pd.DataFrame(bakery_temp.Item.values, columns=['Item'], index=dates)
bakery.head(5)
bakery_daily = bakery.resample('D').count()
bakery_daily.head(5)
bakery_daily['is_local_event'] = np.where(bakery_daily.index.isin(list_of_event),1,0)

bakery_daily['is_bank_holiday'] = np.where(bakery_daily.index.isin(is_bank_holiday),1,0)
bakery_daily['is_local_event'].sum()
bakery_daily['is_bank_holiday'].sum()
bakery_daily.Item.plot()
bakery_daily.boxplot()
bakery_daily.describe()
bakery_daily[bakery_daily['Item'] == 1]
bakery_daily.loc[bakery_daily['Item'] == 1, 'Item'] = 0
bakery_daily[bakery_daily['Item'] == 1]
bakery_daily['Item'].hist()
decomposition = sm.tsa.seasonal_decompose(bakery_daily,model="additive")



fig, ax = plt.subplots()

ax.grid(True)



year = mdates.YearLocator(month=1)

month = mdates.MonthLocator(interval=1)



year_format = mdates.DateFormatter('%Y')

month_format = mdates.DateFormatter('%Y-%m')



ax.xaxis.set_minor_locator(month)

ax.xaxis.grid(True, which = 'minor')

ax.xaxis.set_major_locator(month)

ax.xaxis.set_major_formatter(month_format)



plt.plot(bakery_daily.index, bakery_daily['Item'], c='blue')

plt.plot(decomposition.trend.index, decomposition.trend, c='red')
bakery_daily['Date'] = bakery_daily.index
bakery_daily['Week_Day'] = bakery_daily.Date.dt.day_name()
bakery_daily.head(2)
#train-test-split, fitting-model, testing definition

def fit_linear_model(X, y, day, model_inst):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

    result_list = []



    model_inst.fit(X_train, y_train)

    

    y_pred = model_inst.predict(X_train)

    

    MAE = metrics.mean_absolute_error(y_train, y_pred)

    RMSE =np.sqrt(metrics.mean_squared_error(y_train, y_pred))



    result_list= [day, MAE, RMSE, model_inst.intercept_, model_inst, model_inst.coef_]

    return result_list

# list containing the days of the week

days_of_the_week = ['Week_Day_Monday','Week_Day_Tuesday','Week_Day_Wednesday','Week_Day_Thursday','Week_Day_Friday', 'Week_Day_Saturday','Week_Day_Sunday']
#Creating dataframe to save resuls

columns = ['Day','MAE','RMSE','Intercept','Model', 'Coef']



results_dropping_day = pd.DataFrame(index=range(21), columns=columns)

model_list =[LinearRegression(), Ridge(), Lasso()]



for i, model_item in enumerate(model_list):

    if i == 0:

        n = 0

    elif i == 1:

        n = 7

    else:

        n = 14

            

            

    for i, day in enumerate(days_of_the_week):

        bakery_data_for_model_drop = pd.get_dummies(bakery_daily)

        # set y

        y = bakery_data_for_model_drop['Item']

        del bakery_data_for_model_drop[day]

        del bakery_data_for_model_drop['Item']

        del bakery_data_for_model_drop['Date']



        X = bakery_data_for_model_drop



        results_dropping_day.loc[i + n] = fit_linear_model(X, y, day, model_item)

#Linear regression results

results_dropping_day
bakery_data_for_model = pd.get_dummies(bakery_daily)
bakery_data_for_model.head(2)
del bakery_data_for_model['Week_Day_Wednesday']
bakery_data_for_model.head(2)
bakery_data_for_model.to_pickle('bakery_data_for_model.pkl')
bakery_data_for_model.columns
bakery_data_for_model['Item'].sort_values(ascending=True).head(5)
X = bakery_data_for_model[['is_local_event','is_bank_holiday','Week_Day_Friday', 'Week_Day_Monday',

       'Week_Day_Saturday', 'Week_Day_Sunday', 'Week_Day_Thursday',

       'Week_Day_Tuesday']]

y = bakery_data_for_model['Item']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)



#transform the targe to get a simplier relation, in this case is log(x + 1) becuase there are zeros in data

y_train_log = np.log(y_train + 1)

y_test_log =  np.log(y_test + 1)



print(len(X_train), len(X_test))

print(len(y_train), len(y_test))

# Setup our GridSearch Parmaters

search_parameters = {

    'fit_intercept':  [True, False], 

    'normalize':      [False, True]

}



# Intialize a blank model object

lm = LinearRegression()



# Initialize gridsearch

estimator = GridSearchCV(

    lm, # estimator

    search_parameters, # hyper-parameter space to search 

    cv=5, # number of folds

    scoring="neg_mean_squared_error", # scoring metric to optimise for

    return_train_score=True,

    iid=True

)



# Fit some data

results = estimator.fit(X_train,y_train)

print (results.best_estimator_)
testing = pd.DataFrame(results.cv_results_)
testing
lr = LinearRegression(fit_intercept=True, normalize = False)

lr_log = LinearRegression()



#estimation the line

lr.fit(X_train, y_train)

lr_log.fit(X_train, y_train_log)
print(lr.intercept_, [z for z in zip(X.columns, lr.coef_)])
y_pred = lr.predict(X_train)

y_pred_log = lr_log.predict(X_train)
grid = GridSearchCV(estimator=Ridge(),

                    param_grid={'alpha': np.logspace(-10,10, 20)},

                    scoring='neg_mean_squared_error',

                    return_train_score=True,

                    cv=5, 

                   iid=True)



grid.fit(X_train,y_train)
print(np.sqrt(-grid.best_score_), grid.best_params_)



best_model = grid.best_estimator_

np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
#really small alpha

grid.best_params_
ridge = Ridge(alpha=1e-10, normalize=False)

ridge.fit(X_train, y_train)



print(lr.coef_, np.sqrt(mean_squared_error(y_train, lr.predict(X_train))), "\n")

print(ridge.coef_, np.sqrt(mean_squared_error(y_train, ridge.predict(X_train))))
predictors = X_train.columns



coefRidge = Series(ridge.coef_,predictors).sort_values()



coefRidge.plot(kind='bar', title='Coeficient with Ridge', grid=True)
lasso = Lasso(normalize=False)



lasso.fit(X_train, y_train)
predictors = X_train.columns



coefLasso = Series(lasso.coef_,predictors).sort_values()



coefLasso.plot(kind='bar', title='Coeficient with Lasso', grid=True)
#Get the RMSE and score of all models in a dataframe

RMSE_SCORES_results = pd.DataFrame(columns=['Model', 'Train_RMSE', 'Test_RMSE', 'Train_Score','Test_Score'], index=range(3))

model_list = [lr,ridge,lasso]

y_pred_mean_train = [y_train.mean()] * len(y_train)

y_pred_mean_test = [y_test.mean()] * len(y_test)



for i, item in enumerate(model_list):

    rmse_train = np.sqrt(mean_squared_error(y_train, item.predict(X_train)))

    rmse_test = np.sqrt(mean_squared_error(y_test, item.predict(X_test)))

    score_train = item.score(X_train, y_train)

    score_test = item.score(X_test, y_test)

    item_str = str(item)

    RMSE_SCORES_results.loc[i] = [item_str[:6], rmse_train, rmse_test, score_train,score_test]



RMSE_SCORES_results.loc[3] = ['Baseline',  np.sqrt(metrics.mean_squared_error(y_train, y_pred_mean_train)), np.sqrt(metrics.mean_squared_error(y_test, y_pred_mean_test)),0.0, 0.0]



RMSE_SCORES_results.index = RMSE_SCORES_results.Model
RMSE_SCORES_results
fig, axes = plt.subplots(2, 2, figsize=(15,8), sharex=False, sharey=True, squeeze=False)



fig.suptitle('RMSE and Score', fontsize=12)

fig.text(0.06, 0.5, 'Total Item Sold', ha='center', va='center', rotation='vertical')

#fig.text(0.5, 0.04, 'Hours', ha='center', va='center')

RMSE_train_graph = RMSE_SCORES_results['Train_RMSE']

RMSE_train_graph.plot(ax=axes[0][0], grid=True, kind='barh', title='RMSE for train')



RMSE_test_graph = RMSE_SCORES_results['Test_RMSE']

RMSE_test_graph.plot(ax=axes[0][1], grid=True, kind='barh', title='RMSE for test')



score_train_graph = RMSE_SCORES_results['Train_Score']

score_train_graph.plot(ax=axes[1][0], grid=True, kind='barh', title='Score for train')



score_test_graph = RMSE_SCORES_results['Test_Score']

score_test_graph.plot(ax=axes[1][1], grid=True, kind='barh', title='Score for test')



#plt.xticks(rotation=45)

#RMSE_SCORES_results['Train_RMSE'].plot(kind='bar')
def cross_validation_test(model_name, x_data, y_data, scoring_name, n):

    cv_scores = cross_val_score(model_name, x_data, y_data, scoring=scoring_name, cv=n)

    return np.sqrt(-cv_scores), np.sqrt(-cv_scores.mean())
RMSE_SCORES_results = pd.DataFrame(columns=['Model', 'RMSE', 'Average RMSE'], index=range(2))

model_list = [lr,ridge,lasso]

print('Training set')            

for item in model_list:

    model_string = str(item)

    print('Results for ' + model_string[:10])

    print(cross_validation_test(item,  X, y, 'neg_mean_squared_error', 5))

                                            
results_pred_train = pd.DataFrame({'Actual': y_train, 'LR_Pred': lr.predict(X_train), 'LR_log_Pred': np.exp(lr_log.predict(X_train))-1, 'Lasso_Pred': lasso.predict(X_train), 'Ridge_Pred': ridge.predict(X_train)}) 
results_pred_train.describe()
results_pred_train.boxplot()
results_pred_train[['Actual', 'LR_Pred', 'Lasso_Pred', 'Ridge_Pred']].plot(figsize=(15,7), style={'Actual': '-or', 'LR_Pred': '-ob', 'Lasso_Pred': '-oy', 'Ridge_Pred': '-og'}, grid=True)
results_pred_test = pd.DataFrame({'Actual': y_test, 'LR_Pred': lr.predict(X_test),'LR_log_Pred': np.exp(lr_log.predict(X_test))-1,'Lasso_Pred': lasso.predict(X_test), 'Ridge_Pred': ridge.predict(X_test)})  
results_pred_test.boxplot()
results_pred_test[['Actual', 'LR_Pred', 'Lasso_Pred', 'Ridge_Pred']].plot(figsize=(15,7), style={'Actual': '-or', 'LR_Pred': '-ob', 'Lasso_Pred': '-oy', 'Ridge_Pred': '-og'}, grid=True)
results_pred_test.sort_index()
results_pred_test.describe()
results_pred_test.sort_index().head(2)
results_pred_test.to_pickle('results_pred_test.pkl')
results_pred_train.to_pickle('results_pred_train.pkl')