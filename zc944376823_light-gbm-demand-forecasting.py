import pandas as pd

import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

sns.set()

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import statsmodels.api as sm

import xgboost as xgb

import lightgbm as lgb

from sklearn.model_selection import train_test_split



import warnings

# import the_module_that_warns



warnings.filterwarnings("ignore")



from fbprophet import Prophet
# Input data files are available in the "../input/" directory.

# First let us load the datasets into different Dataframes

def load_data(datapath):

    data = pd.read_csv(datapath)

   # Dimensions

    print('Shape:', data.shape)

    # Set of features we have are: date, store, and item

    display(data.sample(10))

    return data

    

    

train_df = load_data('../input/demand-forecasting-kernels-only/train.csv')

test_df = load_data('../input/demand-forecasting-kernels-only/test.csv')

sample_df = load_data('../input/demand-forecasting-kernels-only/sample_submission.csv')
# Sales distribution across the train data

def sales_dist(data):

    """

        Sales_dist used for Checing Sales Distribution.

        data :  contain data frame which contain sales data

    """

    sales_df = data.copy(deep=True)

    sales_df['sales_bins'] = pd.cut(sales_df.sales, [0, 50, 100, 150, 200, 250])

    print('Max sale:', sales_df.sales.max())

    print('Min sale:', sales_df.sales.min())

    print('Avg sale:', sales_df.sales.mean())

    print()

    return sales_df



sales_df = sales_dist(train_df)



# Total number of data points

total_points = pd.value_counts(sales_df.sales_bins).sum()

print('Sales bucket v/s Total percentage:')

display(pd.value_counts(sales_df.sales_bins).apply(lambda s: (s/total_points)*100))
# Let us visualize the same

sales_count = pd.value_counts(sales_df.sales_bins)

sales_count.sort_values(ascending=True).plot(kind='barh', title='Sales distribution', );

# sns.countplot(sales_count)
# Let us understand the sales data distribution across the stores

def sales_data_understanding(data):    

    store_df = data.copy()

    plt.figure(figsize=(20,10))

    sales_pivoted_df = pd.pivot_table(store_df, index='store', values=['sales','date'], columns='item', aggfunc=np.mean)

    sales_pivoted_df.plot(kind="hist",figsize=(20,10))

    # Pivoted dataframe

    display(sales_pivoted_df)

    return (store_df,sales_pivoted_df)



store_df,sales_pivoted_df = sales_data_understanding(train_df)
# Let us calculate the average sales of all the items by each store

sales_across_store_df = sales_pivoted_df.copy()

sales_across_store_df['avg_sale'] = sales_across_store_df.apply(lambda r: r.mean(), axis=1)
# Scatter plot of average sales per store

sales_store_data = go.Scatter(

    y = sales_across_store_df.avg_sale.values,

    mode='markers',

    marker=dict(

        size = sales_across_store_df.avg_sale.values,

        color = sales_across_store_df.avg_sale.values,

        colorscale='Viridis',

        showscale=True

    ),

    text = sales_across_store_df.index.values

)

data = [sales_store_data]



sales_store_layout = go.Layout(

    autosize= True,

    title= 'Scatter plot of avg sales per store',

    hovermode= 'closest',

    xaxis= dict(

        title= 'Stores',

        ticklen= 10,

        zeroline= False,

        gridwidth= 1,

    ),

    yaxis=dict(

        title= 'Avg Sales',

        ticklen= 10,

        zeroline= False,

        gridwidth= 1,

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=sales_store_layout)

py.iplot(fig,filename='scatter_sales_store')
def sales_insight(sales_pivoted_df):

    # Let us calculate the average sales of each of the item across all the stores

    sales_across_item_df = sales_pivoted_df.copy()

    # Aggregate the sales per item and add it as a new row in the same dataframe

    sales_across_item_df.loc[11] = sales_across_item_df.apply(lambda r: r.mean(), axis=0)

    # Note the 11th index row, which is the average sale of each of the item across all the stores

    #display(sales_across_item_df.loc[11:])

    avg_sales_per_item_across_stores_df = pd.DataFrame(data=[[i+1,a] for i,a in enumerate(sales_across_item_df.loc[11:].values[0])], columns=['item', 'avg_sale'])

    # And finally, sort by avg sale

    avg_sales_per_item_across_stores_df.sort_values(by='avg_sale', ascending=False, inplace=True)

    # Display the top 10 rows

    display(avg_sales_per_item_across_stores_df.head())

    return (sales_across_item_df,avg_sales_per_item_across_stores_df)



sales_across_item_df,avg_sales_per_item_across_stores_df = sales_insight(sales_pivoted_df)
avg_sales_per_item_across_stores_sorted = avg_sales_per_item_across_stores_df.avg_sale.values

# Scatter plot of average sales per item

sales_item_data = go.Bar(

    x=[i for i in range(0, 50)],

    y=avg_sales_per_item_across_stores_sorted,

    marker=dict(

        color=avg_sales_per_item_across_stores_sorted,

        colorscale='Blackbody',

        showscale=True

    ),

    text = avg_sales_per_item_across_stores_df.item.values

)

data = [sales_item_data]



sales_item_layout = go.Layout(

    autosize= True,

    title= 'Scatter plot of avg sales per item',

    hovermode= 'closest',

    xaxis= dict(

        title= 'Items',

        ticklen= 55,

        zeroline= False,

        gridwidth= 1,

    ),

    yaxis=dict(

        title= 'Avg Sales',

        ticklen= 10,

        zeroline= False,

        gridwidth= 1,

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=sales_item_layout)

py.iplot(fig,filename='scatter_sales_item')
def Time_visualization(data):

    store_item_df = data.copy()

    # First, let us filterout the required data

    store_id = 10   # Some store

    item_id = 40    # Some item

    print('Before filter:', store_item_df.shape)

    store_item_df = store_item_df[store_item_df.store == store_id]

    store_item_df = store_item_df[store_item_df.item == item_id]

    print('After filter:', store_item_df.shape)

    #display(store_item_df.head())



    # Let us plot this now

    store_item_ts_data = [go.Scatter(

        x=store_item_df.date,

        y=store_item_df.sales)]

    py.iplot(store_item_ts_data)

    return store_item_df



store_item_df = Time_visualization(train_df)
def sales_monthly(data):

    multi_store_item_df = data.copy()

    # First, let us filterout the required data

    store_ids = [1, 1, 1, 1]   # Some stores

    item_ids = [10, 20, 30, 40]    # Some items

    print('Before filter:', multi_store_item_df.shape)

    multi_store_item_df = multi_store_item_df[multi_store_item_df.store.isin(store_ids)]

    multi_store_item_df = multi_store_item_df[multi_store_item_df.item.isin(item_ids)]

    print('After filter:', multi_store_item_df.shape)

    #display(multi_store_item_df)

    # TODO Monthly avg sales



    # Let us plot this now

    multi_store_item_ts_data = []

    for st,it in zip(store_ids, item_ids):

        flt = multi_store_item_df[multi_store_item_df.store == st]

        flt = flt[flt.item == it]

        multi_store_item_ts_data.append(go.Scatter(x=flt.date, y=flt.sales, name = "Store:" + str(st) + ",Item:" + str(it)))

    py.iplot(multi_store_item_ts_data)

    return (multi_store_item_df)



multi_store_item_df = sales_monthly(train_df)
def split_data(train_data,test_data):

    train_data['date'] = pd.to_datetime(train_data['date'])

    test_data['date'] = pd.to_datetime(test_data['date'])



    train_data['month'] = train_data['date'].dt.month

    train_data['day'] = train_data['date'].dt.dayofweek

    train_data['year'] = train_data['date'].dt.year



    test_data['month'] = test_data['date'].dt.month

    test_data['day'] = test_data['date'].dt.dayofweek

    test_data['year'] = test_data['date'].dt.year



    col = [i for i in test_data.columns if i not in ['date','id']]

    y = 'sales'

    train_x, test_x, train_y, test_y = train_test_split(train_data[col],train_data[y], test_size=0.2, random_state=2018)

    return (train_x, test_x, train_y, test_y,col)



train_x, test_x, train_y, test_y,col = split_data(train_df,test_df)
train_x.shape,train_y.shape,test_x.shape
# from bayes_opt import BayesianOptimization

# def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.02, output_process=False):

#     # prepare data

#     train_data = lgb.Dataset(data=X, label=y)

#     # parameters

#     def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):

#         params = {'application':'regression_l1','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}

#         params["num_leaves"] = int(round(num_leaves))

#         params['feature_fraction'] = max(min(feature_fraction, 1), 0)

#         params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

#         params['max_depth'] = int(round(max_depth))

#         params['lambda_l1'] = max(lambda_l1, 0)

#         params['lambda_l2'] = max(lambda_l2, 0)

#         params['min_split_gain'] = min_split_gain

#         params['min_child_weight'] = min_child_weight

#         cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])

#         return max(cv_result['auc-mean'])

#     # range 

#     lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),

#                                             'feature_fraction': (0.1, 0.9),

#                                             'bagging_fraction': (0.8, 1),

#                                             'max_depth': (5, 8.99),

#                                             'lambda_l1': (0, 5),

#                                             'lambda_l2': (0, 3),

#                                             'min_split_gain': (0.001, 0.1),

#                                             'min_child_weight': (5, 50)}, random_state=0)

#     # optimize

#     lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    

#     # output optimization process

#     if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")

    

#     # return best parameters

#     return lgbBO.res['max']['max_params']



# opt_params = bayes_parameter_opt_lgb(train_x, train_y, init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=100, learning_rate=0.02)
# opt_params
%%time



def model(train_x,train_y,test_x,test_y,col):

    params = {

        'nthread': 10,

         'max_depth': 5,

#         'max_depth': 9,

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'regression_l1',

        'metric': 'mape', # this is abs(a-e)/max(1,a)

#         'num_leaves': 39,

        'num_leaves': 64,

        'learning_rate': 0.2,

       'feature_fraction': 0.9,

#         'feature_fraction': 0.8108472661400657,

#         'bagging_fraction': 0.9837558288375402,

       'bagging_fraction': 0.8,

        'bagging_freq': 5,

        'lambda_l1': 3.097758978478437,

        'lambda_l2': 2.9482537987198496,

#       'lambda_l1': 0.06,

#       'lambda_l2': 0.1,

        'verbose': 1,

        'min_child_weight': 6.996211413900573,

        'min_split_gain': 0.037310344962162616,

        }

    

    lgb_train = lgb.Dataset(train_x,train_y)

    lgb_valid = lgb.Dataset(test_x,test_y)

    model = lgb.train(params, lgb_train, 3000, valid_sets=[lgb_train, lgb_valid],early_stopping_rounds=50, verbose_eval=50)

    y_test = model.predict(test_df[col])

    return y_test,model
%%time

y_test, model = model(train_x,train_y,test_x,test_y,col)
sample_df['sales'] = y_test

sample_df.to_csv('lgb_bayasian_param.csv', index=False)

sample_df['sales'].head()
def average(df1):

    avg  = df1

    df2 = pd.read_csv("../input/private/sub_val-0.132358565029612.csv")

    avg['sales'] = (df1["sales"]*0.4 + df2["sales"]*0.6)

    return avg



avg = average(sample_df)

avg.to_csv("Submission.csv", index=False)