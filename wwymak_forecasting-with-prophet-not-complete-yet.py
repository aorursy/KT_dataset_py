import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from sklearn.externals import joblib
from sklearn.metrics import m
import seaborn as sns
import matplotlib.pyplot as plt
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr 
import datetime
datetime.datetime.strptime
%matplotlib inline 
 
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')
df_test = pd.read_csv('./data/test.csv.zip')
df_test.head()
df = pd.read_csv('./data/train.csv.zip')
df['datetime'] = pd.to_datetime(df['date'])
df.head()
df.tail()
df_train = df[df['datetime']<=datetime.date(2016,12,31)]  
df_test = df[df['datetime']>datetime.date(2016,12,31)]  
df_test.head()
df1 = df[(df.store == 1 ) & (df.item == 1 )]
plt.plot(df1.datetime, df1.sales)
df2 = pd.DataFrame()
df2[['ds', 'y']] = df1[['date', 'sales']]
df2.head()
train_slice = int(0.9 * len(df2['ds']))                      
train = df2[:train_slice]
test = df2[train_slice-10:]
m = Prophet()
m.fit(df2)
test = test[['ds']]
forecast = m.predict(df2)
# forecast.tail()

fig1 = m.plot(forecast)

fig2=m.plot_components(forecast)
df_cv = cross_validation(m, initial='730 days', period='365 days', horizon = '365 days')
df_cv.head()
df_p = performance_metrics(df_cv)
df_p.head()
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')
df_p.mape.mean()
df_item1 = df[df.item ==1]
df_item1.tail()
groups = df_item1.groupby('store')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.datetime, group.sales, label=name)
ax.legend()

df_item1 = df[df.item ==1]
groups = df_item1.groupby('store')

models = []
historic_forecasts = []
metrics = []
for store, group in groups:
    alldata = pd.DataFrame()
    alldata['ds'] = group.datetime
    alldata['y'] = group.sales
    
    train = alldata[alldata['ds']<=datetime.date(2016,12,31)]  
    test = alldata[alldata['ds']>datetime.date(2016,12,31)]  
    
    m = Prophet()
    m.fit(train)
    models.append(m)
    forecast = m.predict(test)
    historic_forecasts.append(forecast)
    df_cv = cross_validation(m, initial='730 days', period='365 days', horizon = '365 days')
    df_p = performance_metrics(df_cv)
    metrics.append(df_p)
    

joblib.dump(models, 'test_models_item1.pkl')
joblib.dump(historic_forecasts, 'test_historicforecasts_item1.pkl')
joblib.dump(metrics, 'test_metrics_item1.pkl')
test_metrics = []
for store, group in groups:
    alldata = pd.DataFrame()
    alldata['ds'] = group.datetime
    alldata['y'] = group.sales
    test = alldata[alldata['ds']>datetime.date(2016,12,31)]  
    try:
        metric_df = historic_forecasts[store -1].set_index('ds')[['yhat']].join(test.set_index('ds').y).reset_index()
        metric_df.dropna(inplace=True)
        test_metrics.append(metric_df)
    except Exception as e:
        print(e, store)
historic_forecasts[0].head()


# def mape(df):
#     """Mean absolute percent error
#     Parameters
#     ----------
#     df: Cross-validation results dataframe.
#     Returns
#     -------
#     Array of mean absolute percent errors.
#     """
#     ape = np.abs((df['y'] - df['yhat']) / (df['y'] + 0.000000000001))
#     return np.mean(ape)



def smape(df):
    """Symmetric mean absolute percentage error
    Parameters
    ----------
    df: Results dataframe.
    Returns
    -------
    Array of symmetric mean absolute percent errors.
    """
    
    #note: adding in + 0.000000000001 to handle division by zero.
    return np.mean(2 * np.abs(df['y'] - df['yhat']) /(df['y'] + df['yhat']+ 0.000000000001))
    

for m in test_metrics:
    print(smape(m))

groups_avg = df_item1.groupby('date').mean()
groups_avg = groups_avg.reset_index()
groups_avg

prophet_input = pd.DataFrame()
prophet_input['ds'] = pd.to_datetime(groups_avg['date'])
prophet_input['y'] = groups_avg['sales']


train_avg = prophet_input[prophet_input['ds']<=datetime.date(2016,12,31)]  
test_avg = test_metrics 

    
avg_model = Prophet()
avg_model.fit(train)

historic_forecasts_avgmodel = []
test_metrics_avgmodel = []
for store, group in groups:
    alldata = pd.DataFrame()
    alldata['ds'] = group.datetime
    alldata['y'] = group.sales
    
    train = alldata[alldata['ds']<=datetime.date(2016,12,31)]  
    test = alldata[alldata['ds']>datetime.date(2016,12,31)]  

    forecast = avg_model.predict(test)
    historic_forecasts_avgmodel.append(forecast)
    
    try:
        metric_df = forecast.set_index('ds')[['yhat']].join(test.set_index('ds').y).reset_index()
        metric_df.dropna(inplace=True)
        test_metrics_avgmodel.append(metric_df)
    except Exception as e:
        print(e, store)

    

for m in test_metrics_avgmodel:
    print(smape(m))
groupby