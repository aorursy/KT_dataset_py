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
from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)

import pandas as pd

from datetime import datetime

import matplotlib.pyplot as plt

import cufflinks as cf

import math

import numpy as np

from statsmodels.tsa.api import VAR

from statsmodels.tsa.stattools import adfuller

from statsmodels.tools.eval_measures import rmse, aic

import warnings

warnings. filterwarnings('ignore')
ndf=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

mdfw=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
def parser(x):

    try:

        return datetime.strptime(x,'%Y-%m-%d')

    except:

        return datetime.strptime(x,' %Y-%m-%d')

for i in range(0,len(ndf)):

   ndf.Date[i]=parser(ndf.Date[i])

for i in range(0,len(mdfw)):

   mdfw.Date[i]=parser(mdfw.Date[i])
pdfw=ndf[ndf['Province_State'].isna()==True]

countries=list(pdfw.Country_Region.unique())

mdfw.head()

t1=mdfw[['Country_Region','ConfirmedCases','Date']]





t1.index.name=''

p1=pd.DataFrame(t1[t1['Country_Region']=='Afghanistan']).set_index('Date').loc['2020-03-28']

p=pd.DataFrame(p1).T

i='Afghanistan'

for i in countries:

      if i not in ['Denmark', 'France', 'Netherlands', 'United Kingdom']:      

        t1=mdfw[['Country_Region','ConfirmedCases','Date']]



        t1.index.name=''

        t1=t1[t1['Country_Region']==i]

        w1=t1[t1['Date']>=datetime.strptime('2020-03-19', '%Y-%m-%d') ]

        w1=w1[t1['Date']<=datetime.strptime('2020-03-28', '%Y-%m-%d')].set_index('Date')

        w1.index.name=''

        

        

        cc1=ndf[ndf['Country_Region']==i].set_index('Date')



        cc=mdfw[mdfw['Country_Region']==i].drop(['Country_Region','Province_State'],axis=1).set_index('Date').drop('Fatalities',axis=1)

    

        cc.index.name=''

        cc['EverydayCC']=cc.ConfirmedCases.diff()

        cc.EverydayCC.iloc[0]=0.0

        

        try:

            model = VAR(endog=cc.drop('Id',axis=1))

            model_fit = model.fit()



            yhat = model_fit.forecast(model_fit.y, steps=33)



            final=pd.DataFrame(data=yhat,index=pd.date_range(start='29/3/2020', periods=33),columns=['ConfirmedCases','EverydayCC'])

    

            final.drop('ConfirmedCases',axis=1)

            final['ConfirmedCases']=final['EverydayCC'].apply(lambda x: cc.iloc[-1][1]+x)

    

            final.drop('EverydayCC',axis=1,inplace=True)

            final['Country_Region']=i

            w1=w1.append(final)

            w1['Province_State']=cc1['Province_State']

            w1['ForecastId']=cc1['ForecastId']

            p =p.append(w1)

        except:

            print('Exception')

        

      else:

           

        

            kkk=mdfw[mdfw['Country_Region']==i ]

            lol=kkk[kkk.Province_State.isna()]

            t1=lol[['Country_Region','ConfirmedCases','Date']]



            t1.index.name=''

            o1=pd.DataFrame(t1[t1['Country_Region']==i]).set_index('Date')

            o1.index.name=''

            t1=o1.loc['2020-03-19':'2020-03-28']

            cc2=ndf[ndf['Country_Region']==i].set_index('Date')

            cc1=cc2[cc2['Province_State'].isna()]

            cc=lol[lol['Country_Region']==i].drop(['Country_Region','Province_State'],axis=1).set_index('Date').drop('Fatalities',axis=1)               

            cc.index.name=''

            cc['EverydayCC']=cc.ConfirmedCases.diff()

            cc.EverydayCC.iloc[0]=0.0

            model = VAR(endog=cc.drop('Id',axis=1))

            model_fit = model.fit()

            yhat = model_fit.forecast(model_fit.y, steps=33)



            final=pd.DataFrame(data=yhat,index=pd.date_range(start='29/3/2020', periods=33),columns=['ConfirmedCases','EverydayCC'])

            final.drop('ConfirmedCases',axis=1)

            final['ConfirmedCases']=final['EverydayCC'].apply(lambda x: cc.iloc[-1][1]+x)    

            final.drop('EverydayCC',axis=1,inplace=True)

            final['Country_Region']=i

            t1=t1.append(final) 

            t1['Country_Region']=cc1['Country_Region']

            t1['ForecastId']=cc1['ForecastId']

            p =p.append(t1)

            

                

    




dp1=p[1:]
t1=mdfw[['Country_Region','Fatalities','Date']]



t1.index.name=''

t1=pd.DataFrame(t1[t1['Country_Region']=='Afghanistan'].iloc[0]).T

p=t1.set_index('Date')

for i in countries:

      if i not in ['Denmark', 'France', 'Netherlands', 'United Kingdom']:      

        t1=mdfw[['Country_Region','Fatalities','Date']]



        t1.index.name=''

        t1=t1[t1['Country_Region']==i]

        w1=t1[t1['Date']>=datetime.strptime('2020-03-19', '%Y-%m-%d') ]

        w1=w1[t1['Date']<=datetime.strptime('2020-03-28', '%Y-%m-%d')].set_index('Date')

        w1.index.name=''

        

        

        cc1=ndf[ndf['Country_Region']==i].set_index('Date')



        cc=mdfw[mdfw['Country_Region']==i].drop(['Country_Region','Province_State'],axis=1).set_index('Date').drop('ConfirmedCases',axis=1)

    

        cc.index.name=''

        cc['EverydayCC']=cc.Fatalities.diff()

        cc.EverydayCC.iloc[0]=0.0

        

        try:

            model = VAR(endog=cc.drop('Id',axis=1))

            model_fit = model.fit()



            yhat = model_fit.forecast(model_fit.y, steps=33)



            final=pd.DataFrame(data=yhat,index=pd.date_range(start='29/3/2020', periods=33),columns=['Fatalities','EverydayCC'])

    

            final.drop('Fatalities',axis=1)

            final['Fatalities']=final['EverydayCC'].apply(lambda x: cc.iloc[-1][1]+x)

    

            final.drop('EverydayCC',axis=1,inplace=True)

            final['Country_Region']=i

            w1=w1.append(final)

            w1['Province_State']=cc1['Province_State']

            w1['ForecastId']=cc1['ForecastId']

            p =p.append(w1)

        except:

            print('Exception')

        

      else:

           

        

            kkk=mdfw[mdfw['Country_Region']==i ]

            lol=kkk[kkk.Province_State.isna()]

            t1=lol[['Country_Region','Fatalities','Date']]



            t1.index.name=''

            o1=pd.DataFrame(t1[t1['Country_Region']==i]).set_index('Date')

            o1.index.name=''

            t1=o1.loc['2020-03-19':'2020-03-28']

            cc2=ndf[ndf['Country_Region']==i].set_index('Date')

            cc1=cc2[cc2['Province_State'].isna()]

            cc=lol[lol['Country_Region']==i].drop(['Country_Region','Province_State'],axis=1).set_index('Date').drop('ConfirmedCases',axis=1)               

            cc.index.name=''

            cc['EverydayCC']=cc.Fatalities.diff()

            cc.EverydayCC.iloc[0]=0.0

            model = VAR(endog=cc.drop('Id',axis=1))

            model_fit = model.fit()

            yhat = model_fit.forecast(model_fit.y, steps=33)



            final=pd.DataFrame(data=yhat,index=pd.date_range(start='29/3/2020', periods=33),columns=['Fatalities','EverydayCC'])

            final.drop('Fatalities',axis=1)

            final['Fatalities']=final['EverydayCC'].apply(lambda x: cc.iloc[-1][1]+x)    

            final.drop('EverydayCC',axis=1,inplace=True)

            final['Country_Region']=i

            t1=t1.append(final) 

            t1['Country_Region']=cc1['Country_Region']

            t1['ForecastId']=cc1['ForecastId']

            p =p.append(t1)

            

                

    


dp2=p[1:]
ndfw=ndf[ndf['Province_State'].isna()==False]

pro=list(ndfw.Province_State.unique())



t1=mdfw[['Province_State','ConfirmedCases','Date']]



t1.index.name=''

t1=t1[t1['Province_State']=='Australian Capital Territory'].iloc[-1]

p=pd.DataFrame(t1).T.set_index('Date')

p.index.name=''
for i in pro:

    

    

        t1=mdfw[['Province_State','ConfirmedCases','Date']]



        t1.index.name=''

        t1=t1[t1['Province_State']==i]

        w1=t1[t1['Date']>=datetime.strptime('2020-03-19', '%Y-%m-%d') ]

        w1=w1[t1['Date']<=datetime.strptime('2020-03-28', '%Y-%m-%d')].set_index('Date')

        w1.index.name=''

        

        

        cc1=ndf[ndf['Province_State']==i].set_index('Date')



        cc=mdfw[mdfw['Province_State']==i].drop(['Country_Region','Province_State'],axis=1).set_index('Date').drop('Fatalities',axis=1)

    

        cc.index.name=''

        cc['EverydayCC']=cc.ConfirmedCases.diff()

        cc.EverydayCC.iloc[0]=0.0

        

        try:

            model = VAR(endog=cc.drop('Id',axis=1))

            model_fit = model.fit()



            yhat = model_fit.forecast(model_fit.y, steps=33)



            final=pd.DataFrame(data=yhat,index=pd.date_range(start='29/3/2020', periods=33),columns=['ConfirmedCases','EverydayCC'])

    

            final.drop('ConfirmedCases',axis=1)

            final['ConfirmedCases']=final['EverydayCC'].apply(lambda x: cc.iloc[-1][1]+x)

    

            final.drop('EverydayCC',axis=1,inplace=True)

            final['Province_State']=i

            w1=w1.append(final)

            w1['Country_Region']=cc1['Country_Region']

            w1['ForecastId']=cc1['ForecastId']

            p =p.append(w1)

        except:

            print('Exception')


dp3=p[1:]

t1=mdfw[['Province_State','Fatalities','Date']]







t1.index.name=''

t1=t1[t1['Province_State']=='Australian Capital Territory'].iloc[-1]

p=pd.DataFrame(t1).T.set_index('Date')

p.index.name=''
for i in pro:

    

    

        t1=mdfw[['Province_State','Fatalities','Date']]



        t1.index.name=''

        t1=t1[t1['Province_State']==i]

        w1=t1[t1['Date']>=datetime.strptime('2020-03-19', '%Y-%m-%d') ]

        w1=w1[t1['Date']<=datetime.strptime('2020-03-28', '%Y-%m-%d')].set_index('Date')

        w1.index.name=''

        

        

        cc1=ndf[ndf['Province_State']==i].set_index('Date')



        cc=mdfw[mdfw['Province_State']==i].drop(['Country_Region','Province_State'],axis=1).set_index('Date').drop('ConfirmedCases',axis=1)

    

        cc.index.name=''

        cc['EverydayCC']=cc.Fatalities.diff()

        cc.EverydayCC.iloc[0]=0.0

        

        try:

            model = VAR(endog=cc.drop('Id',axis=1))

            model_fit = model.fit()



            yhat = model_fit.forecast(model_fit.y, steps=33)



            final=pd.DataFrame(data=yhat,index=pd.date_range(start='29/3/2020', periods=33),columns=['Fatalities','EverydayCC'])

    

            final.drop('Fatalities',axis=1)

            final['Fatalities']=final['EverydayCC'].apply(lambda x: cc.iloc[-1][1]+x)

    

            final.drop('EverydayCC',axis=1,inplace=True)

            final['Province_State']=i

            w1=w1.append(final)

            w1['Country_Region']=cc1['Country_Region']

            w1['ForecastId']=cc1['ForecastId']

            p =p.append(w1)

        except:

            print('Exception')


dp4=p[1:]
dp1=dp1.reset_index()

dp2=dp2.reset_index()

dp5=pd.merge(dp1,dp2,on=['Country_Region','Province_State','ForecastId','index'])

dp3=dp3.reset_index()

dp4=dp4.reset_index()

dp6=pd.merge(dp3,dp4,on=['Country_Region','Province_State','ForecastId','index'])

dp7=dp5.append(dp6)

dp8=dp7.drop(['index','Country_Region','Province_State'],axis=1).sort_values('ForecastId')
dp8['ForecastId']=dp8['ForecastId'].apply(lambda x: int(x))

dp8['a']=dp8['ConfirmedCases']

dp8['ConfirmedCases']=dp8['ForecastId']

dp8['ForecastId']=dp8['a']



dp8.drop('a',axis=1,inplace=True)



dp8.columns=['ForecastId','ConfirmedCases','Fatalities']

dp8.to_csv('submission.csv',index=False)
submission=dp8

print(submission)