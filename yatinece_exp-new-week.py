# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

submission = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')
test['Date']=test.Date.astype('datetime64[ns]')
train['key']=train['Province_State'].astype('str')+ " " + train['Country_Region'].astype('str')



test['key']=test['Province_State'].astype('str')+ " " + test['Country_Region'].astype('str')



train.describe()
train=train[train['Date']<='2020-03-19']
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



def RMSLE(pred,actual):

    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))

train_lag_1=train.groupby(['key']).shift(periods=1)

train_lag_2=train.groupby(['key']).shift(periods=2)

train_lag_3=train.groupby(['key']).shift(periods=3)

train['lag_1_ConfirmedCases']=train_lag_1['ConfirmedCases']

train['lag_1_Fatalities']=train_lag_1['Fatalities']

train['lag_2_ConfirmedCases']=train_lag_2['ConfirmedCases']

train['lag_2_Fatalities']=train_lag_2['Fatalities']

train['lag_3_ConfirmedCases']=train_lag_3['ConfirmedCases']

train['lag_3_Fatalities']=train_lag_3['Fatalities']

train
def pred_ets(fcastperiod,fcastperiod1,actual,ffcast,type_ck='ConfirmedCases',verbose=False):

    

    actual=actual[actual[type_ck]>0]

    index=pd.date_range(start=ffcast.index[0], end=ffcast.index[-1], freq='D')

    data=ffcast[type_ck].values

    ffcast1 = pd.Series(data, index)

    index=pd.date_range(start=actual.index[0], end=actual.index[-1], freq='D')

    data=actual[type_ck].values

    daily_analysis_dat = pd.Series(data, index)

    livestock2=daily_analysis_dat

    fit=[]

    fcast=[]

    fname=[]

    try:

        fit1 = SimpleExpSmoothing(livestock2).fit()

        fcast1 = fit1.forecast(fcastperiod1).rename("SES")

        fit.append(fit1)

        fcast.append(fcast1)

        fname.append('SES')

    except:

        1==1

    try:

        fit2 = Holt(livestock2).fit()

        fcast2 = fit2.forecast(fcastperiod1).rename("Holt")

        fit.append(fit2)

        fcast.append(fcast2)

        fname.append('Holt')

    except:

        1==1

    try:

        fit3 = Holt(livestock2, exponential=True).fit()

        fcast3 = fit3.forecast(fcastperiod1).rename("Exponential")

        fit.append(fit3)

        fcast.append(fcast3)

        fname.append('Exponential')

    except:

        1==1

    try:

        fit4 = Holt(livestock2, damped=True).fit(damping_slope=0.98)

        fcast4 = fit4.forecast(fcastperiod1).rename("AdditiveDamped")

        fit.append(fit4)

        fcast.append(fcast4)

        fname.append('AdditiveDamped')

    except:

        1==1

    try:

        fit5 = Holt(livestock2, exponential=True, damped=True).fit()

        fcast5 = fit5.forecast(fcastperiod1).rename("MultiplicativeDamped")

        fit.append(fit5)

        fcast.append(fcast5)

        fname.append('MultiplicativeDamped')

    except:

        1==1

    try:

        fit6 = Holt(livestock2, damped=True).fit()

        fcast6 = fit6.forecast(fcastperiod1).rename("AdditiveDampedC")

        fit.append(fit6)

        fcast.append(fcast6)

        fname.append('AdditiveDampedC')

    except:

        1==1





    def RMSLE(pred,actual):

        return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))

    

    pred_all_result=pd.concat([pd.DataFrame(k.fittedvalues) for k in fit],axis=1)

    pred_all_result.columns=fname

    all_result=pd.concat([pd.DataFrame(k) for k in fcast],axis=1)

    col_chk=[]

    vvvl=ffcast[type_ck].values.shape[0]

    for k in all_result.columns:

        if verbose: print("actual value for method %s  is = %s" % (k,RMSLE(all_result[k].values,ffcast[type_ck].values)))

        if RMSLE(all_result[k].values[:vvvl],ffcast[type_ck].values) is not np.nan:

            col_chk.append(k)

    col_chk_f=[]

    min_acc=-1

    for k in col_chk:

        acc=RMSLE(pred_all_result[k].values,actual[type_ck].values)

        #if k =='Exponential' and acc>0.01:

                #acc=acc-0.01

        if verbose: print("pred value for method %s  is = %s" % (k,acc))

        if acc is not np.nan:

            col_chk_f.append(k)

            if min_acc==-1:

                min_acc=acc

                model_select=k

            elif acc<min_acc:

                min_acc=acc

                model_select=k

    all_result=all_result.append(pred_all_result,sort=False)



    all_result['best_model']=model_select

    all_result['best_pred']=all_result[model_select]

    return all_result
import sys

orig_stdout = sys.stdout



Fatalities_all_result_final=pd.DataFrame()

ConfirmedCases_all_result_Final=pd.DataFrame()

for keys in train['key'].unique():

    chk=train[train['key']==keys]

    chk.index=chk.Date

    fcastperiod=0

    fcastperiod1=35

    actual=chk[:chk.shape[0]-fcastperiod]

    ffcast=chk[chk.shape[0]-fcastperiod-1:]

    ffcast

    try:

        Fatalities_all_result_1=pred_ets(fcastperiod,fcastperiod1,actual,ffcast,'Fatalities').reset_index()

        

        

    except:

        Fatalities_all_result_1=pd.DataFrame(pd.date_range(start=chk.Date.min(), periods=60+fcastperiod1+1, freq='D')[1:])

        Fatalities_all_result_1.columns=['index']

        Fatalities_all_result_1['best_model']='naive'

        Fatalities_all_result_1['best_pred']=0

        

    Fatalities_all_result_1['key']=keys

    Fatalities_all_result_final=Fatalities_all_result_final.append(Fatalities_all_result_1,sort=True)

    try:

        ConfirmedCases_all_result_1=pred_ets(fcastperiod,fcastperiod1,actual,ffcast,'ConfirmedCases').reset_index()



        

    except:

        ConfirmedCases_all_result_1=pd.DataFrame(pd.date_range(start=train.Date.min(), periods=60+fcastperiod1+1, freq='D')[1:])

        ConfirmedCases_all_result_1.columns=['index']

        ConfirmedCases_all_result_1['best_model']='naive'

        ConfirmedCases_all_result_1['best_pred']=1

    

    ConfirmedCases_all_result_1['key']=keys

    ConfirmedCases_all_result_Final=ConfirmedCases_all_result_Final.append(ConfirmedCases_all_result_1,sort=True)

    print( ' done for %s' % keys)

sys.stdout = orig_stdout
ConfirmedCases_all_result_Final.rename(columns={'index':'Date'},inplace=True)

Fatalities_all_result_final.rename(columns={'index':'Date'},inplace=True)

ConfirmedCases_all_result_Final['best_pred']=np.where(ConfirmedCases_all_result_Final['best_pred'] is np.nan , 0,

                                                       ConfirmedCases_all_result_Final['best_pred'] )

Fatalities_all_result_final['best_pred']=np.where(Fatalities_all_result_final['best_pred'] is np.nan , 0 ,

                                                       Fatalities_all_result_final['best_pred'] )

ConfirmedCases_all_result_Final['best_pred']=np.where(ConfirmedCases_all_result_Final['best_pred'] <0 , 0,

                                                       ConfirmedCases_all_result_Final['best_pred'] )

Fatalities_all_result_final['best_pred']=np.where(Fatalities_all_result_final['best_pred'] <0 , 0 ,

                                                       Fatalities_all_result_final['best_pred'] )

ConfirmedCases_all_result_Final['best_pred_1']=np.where(ConfirmedCases_all_result_Final['AdditiveDamped'] is np.nan , ConfirmedCases_all_result_Final['best_pred'] ,

                                                       ConfirmedCases_all_result_Final['AdditiveDamped'] )

Fatalities_all_result_final['best_pred_1']=np.where(Fatalities_all_result_final['AdditiveDamped'] is np.nan , Fatalities_all_result_final['best_pred'] ,

                                                       Fatalities_all_result_final['AdditiveDamped'] )

ConfirmedCases_all_result_Final['best_pred_1']=np.where(ConfirmedCases_all_result_Final['best_pred'] is np.nan , 0,

                                                       ConfirmedCases_all_result_Final['best_pred'] )

Fatalities_all_result_final['best_pred_1']=np.where(Fatalities_all_result_final['best_pred'] is np.nan , 0 ,

                                                       Fatalities_all_result_final['best_pred'] )

ConfirmedCases_all_result_Final['best_pred_1']=np.where(ConfirmedCases_all_result_Final['best_pred'] <0 , 0,

                                                       ConfirmedCases_all_result_Final['best_pred'] )

Fatalities_all_result_final['best_pred_1']=np.where(Fatalities_all_result_final['best_pred'] <0 , 0 ,

                                                       Fatalities_all_result_final['best_pred'] )
test['Date']=test.Date.astype('datetime64[ns]')
eval1 = ConfirmedCases_all_result_Final[['key','Date','best_pred','best_pred_1']].merge(test, how='right', on=['key','Date'])

eval1.rename(columns={'best_pred':'ConfirmedCases'},inplace=True)

eval1['ConfirmedCases']=eval1['ConfirmedCases'].fillna(0)

eval1
eval2 = Fatalities_all_result_final[['key','Date','best_pred','best_pred_1']].merge(test, how='right', on=['key','Date'])



eval2.rename(columns={'best_pred':'Fatalities'},inplace=True)

eval2['Fatalities']=eval2['Fatalities'].fillna(0)

eval2
sub_prep = eval1[['ForecastId','ConfirmedCases','key']].merge(eval2[['ForecastId','Fatalities']], on=['ForecastId'],  how='left')

sub_prep
sub = sub_prep.merge(submission['ForecastId'], on=['ForecastId'],  how='right')

sub
sub=sub[['ForecastId','ConfirmedCases','Fatalities']]

sub=sub.sort_values('ForecastId')

sub
sub.to_csv('submission.csv',header=['ForecastId','ConfirmedCases','Fatalities'],index=False)
train['Date']=train.Date.astype('datetime64[ns]')

verify=train[['key','Date','ConfirmedCases','Fatalities']].merge(test[['key','Date','ForecastId']], how='inner', on=['key','Date'])

pred=verify[['ForecastId']].merge(sub, how='inner', on=['ForecastId'])
RMSLE(pred['Fatalities'].values,verify['Fatalities'].values)
RMSLE(pred['ConfirmedCases'].values,verify['ConfirmedCases'].values)